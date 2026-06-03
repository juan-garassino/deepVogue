def prepare(**kw):
    raise NotImplementedError(
        "colab.prepare — v2; use local backend or run Colab manually"
    )


def train(**kw):
    raise NotImplementedError(
        "colab.train — v2; run Colab training notebook manually then `make publish`"
    )


def publish(**kw):
    raise NotImplementedError("colab.publish — v2; use local backend")


def project(**kw):
    raise NotImplementedError("colab.project — v2; use local backend or Colab manually")


def walk(**kw):
    raise NotImplementedError(
        "colab.walk — v2; use local backend or HTTP to inference server"
    )


def eval(**kw):
    raise NotImplementedError("colab.eval — v2; use local backend or Colab manually")
