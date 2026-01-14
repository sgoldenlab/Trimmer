import json
import os
from pathlib import Path


class PreferencesManager:
    def __init__(self, config_path=None):
        if config_path is None:
            # Store in user's home directory
            self.config_path = Path.home() / ".trimmer" / "preferences.json"
        else:
            self.config_path = Path(config_path)

        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.preferences = self.load_preferences()

    def get_default_preferences(self):
        return {
            "trial_duration": 120,
            "trial_spacing": 240,
            "last_folder": "",
            "output_folder": "",
            "user_name": "",
            "window_size": [1920, 1080],
            "custom_resolution": None,
        }

    def load_preferences(self):
        if self.config_path.exists():
            try:
                with open(self.config_path, "r") as f:
                    prefs = json.load(f)
                # Merge with defaults to handle new keys
                defaults = self.get_default_preferences()
                defaults.update(prefs)
                return defaults
            except Exception as e:
                print(f"Error loading preferences: {e}")
                return self.get_default_preferences()
        return self.get_default_preferences()

    def save_preferences(self, preferences):
        try:
            with open(self.config_path, "w") as f:
                json.dump(preferences, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving preferences: {e}")
            return False

    def update_preference(self, key, value):
        self.preferences[key] = value
        return self.save_preferences(self.preferences)
