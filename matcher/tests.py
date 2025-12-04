from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
from rest_framework.test import APIClient
from unittest.mock import patch


class SummarizeAudioViewTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.url = "/matcher/summarize/audio"

    @patch("matcher.views.summarize_audio_upload", return_value="file summary")
    def test_multipart_file(self, mock_sum):
        audio = SimpleUploadedFile("sample.webm", b"123", content_type="audio/webm")
        resp = self.client.post(self.url, {"file": audio}, format="multipart")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json().get("summary"), "file summary")
        mock_sum.assert_called_once()

    @patch("matcher.views.summarize_from_url", return_value="url summary")
    def test_form_url_field(self, mock_sum):
        resp = self.client.post(self.url, {"url": "http://example.com/a.webm"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json().get("summary"), "url summary")
        mock_sum.assert_called_once_with("http://example.com/a.webm")

    @patch("matcher.views.summarize_from_url", return_value="json url summary")
    def test_json_url_field(self, mock_sum):
        resp = self.client.post(self.url, {"url": "http://example.com/a.webm"}, format="json")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json().get("summary"), "json url summary")
        mock_sum.assert_called_once_with("http://example.com/a.webm")

    @patch("matcher.views.summarize_from_url", return_value="form urlencoded summary")
    def test_form_urlencoded_body(self, mock_sum):
        resp = self.client.post(
            self.url,
            data="url=http%3A%2F%2Fexample.com%2Fa.webm",
            content_type="application/x-www-form-urlencoded",
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json().get("summary"), "form urlencoded summary")
        mock_sum.assert_called_once_with("http://example.com/a.webm")

    @patch("matcher.views.summarize_from_url", return_value="alias url summary")
    def test_recording_url_alias(self, mock_sum):
        resp = self.client.post(self.url, {"recording_url": "http://example.com/a.webm"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json().get("summary"), "alias url summary")
        mock_sum.assert_called_once_with("http://example.com/a.webm")

    @patch("matcher.views.summarize_from_url", return_value="query url summary")
    def test_url_in_query_params(self, mock_sum):
        resp = self.client.post(f"{self.url}?url=http://example.com/a.webm")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json().get("summary"), "query url summary")
        mock_sum.assert_called_once_with("http://example.com/a.webm")

    def test_empty_request_returns_204(self):
        resp = self.client.post(self.url)
        self.assertEqual(resp.status_code, 204)

    def test_non_empty_body_without_url_returns_400(self):
        resp = self.client.post(
            self.url,
            data="something=present",
            content_type="application/x-www-form-urlencoded",
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("file is required", resp.json().get("error", ""))

    @patch("matcher.views.summarize_audio_upload", return_value="raw bytes summary")
    def test_raw_body_no_multipart(self, mock_sum):
        resp = self.client.post(
            self.url,
            data=b"\x00\x01",
            content_type="application/octet-stream",
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json().get("summary"), "raw bytes summary")
        mock_sum.assert_called_once()
