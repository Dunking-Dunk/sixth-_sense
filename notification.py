import requests


class NotificationSender:
    def __init__(self ,voice_assistant,sub_id="sub-id-1", app_id=19745, app_token="iV4ceRtjBYygvWfLa5Bu3z"):
        self.voice_assistant = voice_assistant
        self.sub_id = sub_id
        self.app_id = app_id
        self.app_token = app_token
        self.url = "https://app.nativenotify.com/api/indie/notification"
        self.headers = {"Content-Type": "application/json"}

    def send_notification(self, title="Alert!! Alert!!", message="Emergency"):
        payload = {
            "subID": self.sub_id,
            "appId": self.app_id,
            "appToken": self.app_token,
            "title": title,
            "message": message
        }

        response = requests.post(self.url, json=payload, headers=self.headers)
        self.voice_assistant.speak("Alert message sent")

# Example usage
# sender = NotificationSender(
#     sub_id="sub-id-1",
#     app_id=19745,
#     app_token="iV4ceRtjBYygvWfLa5Bu3z"
# )

# sender.send_notification(
#     title="Alert!! Alert!!",
#     message="Emergency"
# )