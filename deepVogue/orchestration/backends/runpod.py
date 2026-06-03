def prepare(**kw):
    raise NotImplementedError("runpod.prepare — v2; use local backend")


def train(**kw):
    raise NotImplementedError(
        "runpod.train — v2; submit RunPod job manually then `make publish`"
    )


def publish(**kw):
    raise NotImplementedError("runpod.publish — v2; use local backend")


def project(**kw):
    raise NotImplementedError("runpod.project — v2; use local backend")


def walk(**kw):
    raise NotImplementedError(
        "runpod.walk — v2; use local backend or HTTP to inference server"
    )


def eval(**kw):
    raise NotImplementedError("runpod.eval — v2; use local backend")
