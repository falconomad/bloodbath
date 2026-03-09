import time
from datetime import datetime, timezone
import config
from data_ingestion import market_data, news_feed, calendar_events
from ai_engines import candidate_ranker, executive_board, sentiment_analyst, technical_analyst
from execution import risk_manager, telemetry, trade_executor
from alpaca.trading.client import TradingClient


def _snapshot_account(account_obj, buying_power=None):
    eq = float(getattr(account_obj, "equity", 0.0) or 0.0)
    last_eq = float(getattr(account_obj, "last_equity", eq) or eq)
    return {
        "equity": eq,
        "last_equity": last_eq,
        "buying_power": float(buying_power if buying_power is not None else getattr(account_obj, "buying_power", 0.0) or 0.0),
        "cash": float(getattr(account_obj, "cash", 0.0) or 0.0),
    }


def _snapshot_positions(position_objs):
    rows = []
    for p in (position_objs or []):
        rows.append(
            {
                "symbol": str(getattr(p, "symbol", "")),
                "qty": float(getattr(p, "qty", 0.0) or 0.0),
                "market_value": float(getattr(p, "market_value", 0.0) or 0.0),
                "unrealized_pl_pct": float(getattr(p, "unrealized_plpc", 0.0) or 0.0) * 100,
                "unrealized_pl": float(getattr(p, "unrealized_pl", 0.0) or 0.0),
                "current_price": float(getattr(p, "current_price", 0.0) or 0.0),
            }
        )
    return rows


def _write_profit_summary(trading_client, fallback_account_snapshot, fallback_positions_snapshot, trades_executed, run_status):
    try:
        refreshed_account = trading_client.get_account()
        refreshed_positions = trading_client.get_all_positions()
        account_snapshot = _snapshot_account(refreshed_account)
        positions_snapshot = _snapshot_positions(refreshed_positions)
    except Exception:
        account_snapshot = dict(fallback_account_snapshot or {})
        positions_snapshot = list(fallback_positions_snapshot or [])

    summary = telemetry.update_profit_summary(
        account_snapshot=account_snapshot,
        positions_snapshot=positions_snapshot,
        trades_executed=trades_executed,
        run_status=run_status,
        path=config.PROFIT_SUMMARY_PATH,
        baseline_equity_hint=float(config.BASELINE_EQUITY),
    )
    telemetry.log_event("profit_summary", summary, path=config.ENGINE_EVENTS_PATH)
    print(
        "📘 Profit Summary: "
        f"equity=${summary['equity']:,.2f} | "
        f"total=${summary['total_profit_since_baseline']:,.2f} ({summary['total_return_since_baseline_pct']:.2f}%) | "
        f"daily=${summary['daily_profit']:,.2f} ({summary['daily_profit_pct']:.2f}%) | "
        f"open_unrealized=${summary['unrealized_pl_open_positions']:,.2f}"
    )


def _forced_exit_recommendations(current_positions):
    forced = []
    for p in (current_positions or []):
        symbol = str(p.get("symbol", "")).upper().strip()
        if not symbol:
            continue
        pnl = float(p.get("unrealized_pl_pct", 0.0))
        if pnl > float(config.PROFIT_TAKE_PCT):
            forced.append(
                {
                    "symbol": symbol,
                    "action": "sell",
                    "allocation_pct": 100,
                    "chain_of_thought": f"Deterministic profit-take trigger ({pnl:.2f}% > {config.PROFIT_TAKE_PCT:.2f}%).",
                }
            )
        elif pnl < float(config.STOP_LOSS_PCT):
            forced.append(
                {
                    "symbol": symbol,
                    "action": "sell",
                    "allocation_pct": 100,
                    "chain_of_thought": f"Deterministic stop-loss trigger ({pnl:.2f}% < {config.STOP_LOSS_PCT:.2f}%).",
                }
            )
    return forced


def _should_call_gemini_now():
    step = max(int(config.GEMINI_MINUTE_STEP), 1)
    minute = datetime.now(timezone.utc).minute
    return (minute % step) == 0

def main():
    print("="*50)
    print("🚀 INITIALIZING BLOODBATH MULTI-ENGINE PIPELINE 🚀")
    print("="*50)
    
    trading_client = TradingClient(config.ALPACA_API_KEY, config.ALPACA_API_SECRET, paper=config.PAPER_TRADING)
    
    # ---------------------------------------------------------
    # 1. FETCH MARKET STATE
    # ---------------------------------------------------------
    print("\n[Stage 1] Fetching Broad Market State...")
    try:
        account = trading_client.get_account()
        buying_power = float(account.buying_power)
        print(f"💰 Available Buying Power: ${buying_power:,.2f}")
        clock = trading_client.get_clock()
        market_open = bool(getattr(clock, "is_open", True))
        print(f"🕒 Market Open: {market_open}")
        
        positions = trading_client.get_all_positions()
        current_positions = _snapshot_positions(positions)
        print(f"📊 Open Positions: {len(current_positions)}")
        for pos in current_positions:
            print(f"   - {pos['symbol']}: {pos['unrealized_pl_pct']:.2f}% P/L")
            
    except Exception as e:
        print(f"❌ Failed to fetch account state: {e}")
        return
    account_snapshot_at_start = _snapshot_account(account, buying_power=buying_power)

    forced_exit_recs = _forced_exit_recommendations(current_positions)
    forced_exit_symbols = {str(x.get("symbol", "")).upper() for x in forced_exit_recs}
    if forced_exit_recs:
        print(f"⚠️ Forced exit triggers detected: {', '.join(sorted(forced_exit_symbols))}")

    # ---------------------------------------------------------
    # 2. FETCH TOP MOVERS & HISTORICAL CONTEXT
    # ---------------------------------------------------------
    if config.ONLY_WHEN_MARKET_OPEN and not market_open:
        print("\n[Gate] Market is closed. Skipping new-entry analysis; only forced exits will be processed.")
        top_movers = []
        history = {}
        spy_context = []
        regime = "neutral"
    else:
        print("\n[Stage 2] Scanning for Top Market Movers...")
        top_movers = market_data.get_top_movers(limit=15)
        print(f"🔍 Found {len(top_movers)} volatile movers.")
        
        symbols_to_fetch = ["SPY"] + [m["symbol"] for m in top_movers]
        print(f"📈 Fetching 5-day historical technical data for {len(symbols_to_fetch)} symbols...")
        history = market_data.get_history_context(symbols_to_fetch)
        spy_context = history.pop("SPY", [])
        regime = candidate_ranker.detect_market_regime(spy_context)
        print(f"🧭 Detected market regime: {regime}")
        
        print("\n[Filter] Vetting candidates by liquidity and price...")
        candidates = []
        for mover in top_movers:
            sym = mover["symbol"]
            bars = history.get(sym, [])
            if not bars:
                continue
                
            # Average volume over last 5 days
            avg_vol = sum(b['v'] for b in bars) / len(bars)
            dollar_vol = avg_vol * mover["price"]
            
            if mover["price"] >= config.MIN_PRICE and dollar_vol >= config.MIN_DOLLAR_VOLUME:
                candidates.append((dollar_vol, mover))
                
        # Sort by highest dollar volume to prioritize the most liquid market movers
        candidates.sort(reverse=True, key=lambda x: x[0])
        top_movers = [c[1] for c in candidates[:config.MAX_CANDIDATES]]
        print(f"🎯 Filtered down to {len(top_movers)} absolute best candidates based on ${config.MIN_PRICE}+ price and ${config.MIN_DOLLAR_VOLUME:,.0f}+ 5-day volume.")
    
    if len(top_movers) == 0 and not forced_exit_recs:
        print("🤷 No candidates met liquidity requirements and no forced exits. Ending run.")
        _write_profit_summary(
            trading_client=trading_client,
            fallback_account_snapshot=account_snapshot_at_start,
            fallback_positions_snapshot=current_positions,
            trades_executed=0,
            run_status="no_candidates",
        )
        return

    # ---------------------------------------------------------
    # 3. FETCH NEWS & SENTIMENT CONTEXT
    # ---------------------------------------------------------
    print("\n[Stage 3] Fetching Fundamental Catalysts...")
    mover_symbols = [m["symbol"] for m in top_movers if str(m.get("symbol", "")).upper() not in forced_exit_symbols]
    
    print("📰 Pulling Alpaca News headlines...")
    news_context = news_feed.get_news_context(mover_symbols)
    
    print("🗓️ Pulling Corporate Calendars...")
    upcoming_events = calendar_events.get_earnings_calendar(mover_symbols)
    macro_calendar = calendar_events.get_macro_calendar()

    market_state = {
        "account": {"buying_power": buying_power},
        "open_positions": current_positions,
        "broad_market_spy": spy_context,
        "todays_top_movers": top_movers,
        "market_regime": regime,
    }

    # ---------------------------------------------------------
    # 4. RUN AI ENGINES PER SYMBOL
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("🧠 ENGAGING AI COUNCIL FOR INDIVIDUAL ASSETS 🧠")
    print("="*50)

    if forced_exit_recs:
        telemetry.log_event(
            "forced_exit_candidates",
            {"symbols": sorted(list(forced_exit_symbols)), "count": len(forced_exit_recs)},
            path=config.ENGINE_EVENTS_PATH,
        )

    # Phase A: local deterministic feature extraction and ranking (0 Gemini calls).
    pre_scored = []
    for mover in top_movers:
        symbol = mover["symbol"]
        if str(symbol).upper() in forced_exit_symbols:
            print(f"\n--- Analyzing: {symbol} ---")
            print("   ⏩ Skipping new-entry analysis: symbol already has deterministic forced-exit trigger.")
            continue
        print(f"\n--- Analyzing: {symbol} ---")

        sym_history = history.get(symbol, [])
        sym_news = news_context.get(symbol, [])
        sym_events = upcoming_events.get(symbol, "None")

        if not sym_history:
            print(f"   ⚠️ Skipping {symbol} due to missing price history.")
            continue

        print("   ⚙️ Running local technical + sentiment pre-score...")
        tech_report = technical_analyst.evaluate_technicals(symbol, sym_history, spy_context)
        sent_report = sentiment_analyst.evaluate_sentiment(symbol, sym_news, sym_events, macro_calendar)
        tech_score = int(tech_report.get("score", 0) or 0)
        sent_score = int(sent_report.get("score", 0) or 0)
        sent_conf = float(sent_report.get("confidence", 0.5) or 0.5)
        features = candidate_ranker.build_features(symbol, sym_history, spy_context, sym_news)
        if not features:
            print("      -> rejected: failed to build local features")
            continue
        reject_reasons = candidate_ranker.hard_reject_reasons(features, regime)
        rank = candidate_ranker.rank_candidate(features, tech_score, sent_score, regime)
        print(
            f"      -> Technical: {tech_score}/100 | Sentiment: {sent_score}/100 | "
            f"SentConf: {100.0 * sent_conf:.1f}% | LocalScore: {rank['local_score']:.1f} | "
            f"Confidence: {100.0 * rank['confidence']:.1f}%"
        )
        if reject_reasons:
            print(f"      -> hard reject: {', '.join(reject_reasons)}")

        telemetry.log_event(
            "candidate_scored",
            {
                "symbol": symbol,
                "regime": regime,
                "tech_score": tech_score,
                "sent_score": sent_score,
                "local_score": rank["local_score"],
                "confidence": rank["confidence"],
                "sent_confidence": sent_conf,
                "reject_reasons": reject_reasons,
                "features": features,
                "mover_price": mover.get("price"),
                "mover_change_pct": mover.get("change_pct"),
                "earnings_context": sym_events,
                "macro_context": macro_calendar,
            },
            path=config.ENGINE_EVENTS_PATH,
        )
        pre_scored.append(
            {
                "mover": mover,
                "tech_report": tech_report,
                "sent_report": sent_report,
                "features": features,
                "rank": rank,
                "reject_reasons": reject_reasons,
            }
        )

    if not pre_scored:
        print("\n🤷 No candidates survived local pre-scoring.")
        if forced_exit_recs:
            final_recommendations = list(forced_exit_recs)
        else:
            _write_profit_summary(
                trading_client=trading_client,
                fallback_account_snapshot=account_snapshot_at_start,
                fallback_positions_snapshot=current_positions,
                trades_executed=0,
                run_status="no_prescored_candidates",
            )
            return
    else:
        shortlisted = [
            item for item in pre_scored
            if not item["reject_reasons"]
            and int(item["tech_report"].get("score", 0)) >= config.PRE_FILTER_MIN_TECH_SCORE
            and int(item["sent_report"].get("score", 0)) >= config.PRE_FILTER_MIN_SENTIMENT_SCORE
            and float(item["rank"]["local_score"]) >= float(config.PRE_FILTER_MIN_LOCAL_SCORE)
            and float(item["rank"]["confidence"]) >= float(config.PRE_FILTER_MIN_CONFIDENCE)
        ]
        if not shortlisted:
            relaxed_sent = max(0, int(config.PRE_FILTER_MIN_SENTIMENT_SCORE) - int(config.RELAXED_SENTIMENT_DELTA))
            relaxed_conf = max(0.30, float(config.PRE_FILTER_MIN_CONFIDENCE) - float(config.RELAXED_CONFIDENCE_DELTA))
            relaxed = [
                item for item in pre_scored
                if not item["reject_reasons"]
                and int(item["tech_report"].get("score", 0)) >= config.PRE_FILTER_MIN_TECH_SCORE
                and int(item["sent_report"].get("score", 0)) >= relaxed_sent
                and float(item["rank"]["local_score"]) >= float(config.PRE_FILTER_MIN_LOCAL_SCORE)
                and float(item["rank"]["confidence"]) >= float(relaxed_conf)
            ]
            if not relaxed:
                print(
                    f"\n[PreFilter] No symbols met thresholds "
                    f"(tech>={config.PRE_FILTER_MIN_TECH_SCORE}, sentiment>={config.PRE_FILTER_MIN_SENTIMENT_SCORE}, "
                    f"local_score>={config.PRE_FILTER_MIN_LOCAL_SCORE}, conf>={config.PRE_FILTER_MIN_CONFIDENCE:.2f})."
                )
                shortlisted = []
            else:
                relaxed.sort(key=lambda x: (x["rank"]["local_score"], x["rank"]["confidence"]), reverse=True)
                shortlisted = relaxed[: max(int(config.MAX_RELAXED_CANDIDATES), 1)]
                print(
                    f"\n[PreFilter] Strict gates found 0 candidates. "
                    f"Relaxed fallback selected {len(shortlisted)} symbol(s) "
                    f"(sentiment>={relaxed_sent}, conf>={relaxed_conf:.2f})."
                )
                telemetry.log_event(
                    "shortlist_relaxed_fallback",
                    {
                        "relaxed_sentiment_threshold": relaxed_sent,
                        "relaxed_confidence_threshold": round(float(relaxed_conf), 4),
                        "selected_symbols": [item["mover"]["symbol"] for item in shortlisted],
                    },
                    path=config.ENGINE_EVENTS_PATH,
                )

        shortlisted.sort(key=lambda x: (x["rank"]["local_score"], x["rank"]["confidence"]), reverse=True)
        gemini_budget = max(int(config.MAX_GEMINI_CALLS_PER_RUN), 1)
        selected_for_gemini = shortlisted[:gemini_budget]

        use_gemini_now = _should_call_gemini_now()
        print(
            f"\n[PreFilter] {len(shortlisted)} passed local thresholds. "
            f"Selected {len(selected_for_gemini)} candidate(s). Gemini slot now={use_gemini_now} "
            f"(minute-step={config.GEMINI_MINUTE_STEP})."
        )
        telemetry.log_event(
            "shortlist_selected",
            {
                "regime": regime,
                "selected_symbols": [item["mover"]["symbol"] for item in selected_for_gemini],
                "total_passed": len(shortlisted),
                "gemini_budget": gemini_budget,
                "use_gemini_now": use_gemini_now,
            },
            path=config.ENGINE_EVENTS_PATH,
        )

        final_recommendations = list(forced_exit_recs)
        if use_gemini_now and selected_for_gemini:
            print("   👑 Running Executive Board Synthesis (single Gemini batch call)...")
            batch_payload = [
                {
                    "symbol": item["mover"]["symbol"],
                    "technical_report": item["tech_report"],
                    "sentiment_report": item["sent_report"],
                }
                for item in selected_for_gemini
            ]
            batch_decisions = executive_board.make_batch_decisions(
                candidates=batch_payload,
                current_positions=current_positions,
                buying_power=buying_power,
            )

            for item in selected_for_gemini:
                symbol = item["mover"]["symbol"]
                exec_decision = batch_decisions.get(symbol) or {
                    "symbol": symbol,
                    "action": "hold",
                    "allocation_pct": 0,
                    "chain_of_thought": "No executive decision returned for symbol.",
                }
                print(f"   {symbol} -> Action: {exec_decision.get('action', 'HOLD').upper()} | Alloc: {exec_decision.get('allocation_pct', 0)}%")
                print(f"      Reasoning: {exec_decision.get('chain_of_thought', '')}")
                final_recommendations.append(exec_decision)
                telemetry.log_event(
                    "gemini_decision",
                    {
                        "symbol": symbol,
                        "decision": exec_decision,
                        "local_score": item["rank"]["local_score"],
                        "confidence": item["rank"]["confidence"],
                        "tech_score": item["tech_report"].get("score", 0),
                        "sent_score": item["sent_report"].get("score", 0),
                        "sent_confidence": item["sent_report"].get("confidence", 0.5),
                    },
                    path=config.ENGINE_EVENTS_PATH,
                )
        else:
            print("   ⏭️ Skipping Gemini this cycle (cadence gate). Forced exits only.")
    # if pre_scored was empty and forced exits existed, final_recommendations defined above
    if 'final_recommendations' not in locals():
        final_recommendations = list(forced_exit_recs)

    # ---------------------------------------------------------
    # 5. RISK MANAGEMENT & EXECUTION
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("🛡️ EXECUTING TRADES THROUGH RISK MANAGER 🛡️")
    print("="*50)
    
    trades_executed = 0
    for rec in final_recommendations:
        validated_trade = risk_manager.validate_and_size_trade(rec, market_state)
        telemetry.log_event(
            "risk_validation",
            {
                "symbol": rec.get("symbol"),
                "action": rec.get("action"),
                "allocation_pct": rec.get("allocation_pct"),
                "validated": bool(validated_trade),
                "validated_trade": validated_trade,
            },
            path=config.ENGINE_EVENTS_PATH,
        )
        if validated_trade:
            order = trade_executor.execute_order(validated_trade)
            if order:
                trades_executed += 1
                telemetry.log_event(
                    "order_executed",
                    {
                        "symbol": validated_trade.get("symbol"),
                        "action": validated_trade.get("action"),
                        "qty": validated_trade.get("qty"),
                        "capped_alloc_pct": validated_trade.get("capped_alloc_pct"),
                    },
                    path=config.ENGINE_EVENTS_PATH,
                )
                
    print(f"\n✅ Pipeline Complete. Executed {trades_executed} trades today.")
    _write_profit_summary(
        trading_client=trading_client,
        fallback_account_snapshot=account_snapshot_at_start,
        fallback_positions_snapshot=current_positions,
        trades_executed=trades_executed,
        run_status="completed",
    )

if __name__ == "__main__":
    main()
