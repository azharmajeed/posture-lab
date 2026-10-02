import threading


def notify(message: str):
    """The preview banner always works; OS notification support is best-effort."""
    print(f"Posture reminder: {message}", flush=True)

    def deliver():
        try:
            from plyer import notification

            notification.notify(
                title="Posture Lab",
                message="Time to adjust: " + message,
                app_name="Posture Lab",
                timeout=6,
            )
        except Exception as exc:
            print(f"Desktop notification unavailable ({type(exc).__name__}); see preview banner.")

    threading.Thread(target=deliver, daemon=True).start()
