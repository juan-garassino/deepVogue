"""Local-nano backend: real for prepare/publish, mock for train/project/walk/eval."""

from __future__ import annotations


def prepare(**kw):
    raise NotImplementedError


def train(**kw):
    raise NotImplementedError


def publish(**kw):
    raise NotImplementedError


def project(**kw):
    raise NotImplementedError


def walk(**kw):
    raise NotImplementedError


def eval(**kw):
    raise NotImplementedError
