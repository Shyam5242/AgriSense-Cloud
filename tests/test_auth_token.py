import auth_token as T


def test_valid_token_roundtrip():
    t = T.make_token("secret", ttl=60, now=1000)
    assert T.valid_token("secret", t, now=1030)


def test_expired_token_rejected():
    t = T.make_token("secret", ttl=60, now=1000)
    assert not T.valid_token("secret", t, now=1061)


def test_wrong_password_or_changed_password_rejected():
    t = T.make_token("secret", ttl=60, now=1000)
    assert not T.valid_token("other", t, now=1010)


def test_tampered_or_garbage_tokens_rejected():
    t = T.make_token("secret", ttl=60, now=1000)
    expiry, sig = t.split(".")
    assert not T.valid_token("secret", f"{int(expiry) + 99999}.{sig}", now=1010)   # longer expiry, same signature
    for bad in ["", "abc", "123", "x.y", None, 5, "."]:
        assert not T.valid_token("secret", bad, now=1010)
    assert not T.valid_token("", t, now=1010)
