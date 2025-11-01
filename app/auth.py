import time, jwt, requests
from jwt import algorithms
from django.conf import settings
from rest_framework.authentication import BaseAuthentication
from rest_framework import exceptions
from django.contrib.auth.models import AnonymousUser

class _JWKS:
    cache=None; exp=0
    @classmethod
    def get(cls):
        if cls.cache and cls.exp>time.time(): return cls.cache
        r=requests.get(f"{settings.KEYCLOAK_ISSUER}/protocol/openid-connect/certs", timeout=3)
        r.raise_for_status(); cls.cache=r.json(); cls.exp=time.time()+300; return cls.cache

class KeycloakJWTAuthentication(BaseAuthentication):
    def authenticate(self, request):
        auth=request.headers.get("Authorization","")
        if not auth.startswith("Bearer "): return None
        token=auth.split()[1]
        try:
            kid=jwt.get_unverified_header(token)["kid"]
            key=next(k for k in _JWKS.get()["keys"] if k["kid"]==kid)
            pub=algorithms.RSAAlgorithm.from_jwk(key)
            claims=jwt.decode(token, pub, algorithms=["RS256"],
                              audience=settings.KEYCLOAK_AUDIENCE,
                              issuer=settings.KEYCLOAK_ISSUER)
            return (AnonymousUser(), claims)
        except Exception as e:
            raise exceptions.AuthenticationFailed(str(e))