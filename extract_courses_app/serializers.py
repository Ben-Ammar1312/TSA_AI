from rest_framework import serializers

class ExtractRequestSerializer(serializers.Serializer):
    text = serializers.CharField()
