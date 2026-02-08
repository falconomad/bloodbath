
def detect_events(news):

    keywords = ["earnings", "profit", "acquisition", "investment", "launch"]

    for n in news:
        for k in keywords:
            if k.lower() in n.lower():
                return True

    return False
