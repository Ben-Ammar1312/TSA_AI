from rest_framework import serializers
from .models import SubjectTarget, SubjectAlias
from .utils import normalize_label


class SubjectTargetSerializer(serializers.ModelSerializer):
    class Meta:
        model = SubjectTarget
        fields = '__all__'


class SubjectAliasSerializer(serializers.ModelSerializer):
    # use target code instead of UUID for convenience
    target = serializers.SlugRelatedField(
        slug_field="code",
        queryset=SubjectTarget.objects.all()
    )
    norm_label = serializers.CharField(read_only=True)

    class Meta:
        model = SubjectAlias
        fields = ["id", "target", "label", "norm_label", "language"]

    def validate(self, attrs):
        # keep norm_label consistent with label
        label = attrs.get("label") or getattr(self.instance, "label", "")
        attrs["norm_label"] = normalize_label(label)
        return attrs