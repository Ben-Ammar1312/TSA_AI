from rest_framework import viewsets
from rest_framework.permissions import IsAuthenticated

from .models import SubjectTarget, SubjectAlias
from .serializers import SubjectTargetSerializer, SubjectAliasSerializer


class SubjectTargetViewSet(viewsets.ModelViewSet):
    queryset = SubjectTarget.objects.all()
    serializer_class = SubjectTargetSerializer


class SubjectAliasViewSet(viewsets.ModelViewSet):
    queryset = SubjectAlias.objects.select_related("target").all()
    serializer_class = SubjectAliasSerializer
    permission_classes = [IsAuthenticated]

    # simple filters: ?language=fr&target_code=info.algo.ds&q=tri
    def get_queryset(self):
        qs = super().get_queryset()
        lang = self.request.query_params.get("language")
        tcode = self.request.query_params.get("target_code")
        q = self.request.query_params.get("q")
        if lang:
            qs = qs.filter(language=lang)
        if tcode:
            qs = qs.filter(target__code=tcode)
        if q:
            qs = qs.filter(Q(label__icontains=q) | Q(norm_label__icontains=q))
        return qs

    def perform_create(self, serializer):
        # ensure norm_label is set by serializer.validate()
        serializer.save()

    def perform_update(self, serializer):
        serializer.save()