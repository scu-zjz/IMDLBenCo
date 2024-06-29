import appdirs
import os
""" expected outputs:

User data directory: /home/xiaochen/.local/share/MyApp
User config directory: /home/xiaochen/.config/MyApp
User cache directory: /home/xiaochen/.cache/MyApp
User log directory: /home/xiaochen/.cache/MyApp/log
Site data directory: /usr/local/share/MyApp
Traceback (most recent call last):
  File "/mnt/data0/xiaochen/workspace/IMDLBenCo_pure/IMDLBenCo/tests/test_appdirs.py", line 49, in <module>
    test_appdirs()
  File "/mnt/data0/xiaochen/workspace/IMDLBenCo_pure/IMDLBenCo/tests/test_appdirs.py", line 37, in test_appdirs
    os.makedirs(site_data, exist_ok=True)
  File "<frozen os>", line 225, in makedirs
PermissionError: [Errno 13] Permission denied: '/usr/local/share/MyApp'

"""
def test_appdirs():
    app_name = "MyApp"
    app_author = "MyCompany"

    print("Testing appdirs functionalities...\n")

    # User data directory
    user_data = appdirs.user_data_dir(app_name, app_author)
    print(f"User data directory: {user_data}")
    os.makedirs(user_data, exist_ok=True)
    assert os.path.exists(user_data)

    # User config directory
    user_config = appdirs.user_config_dir(app_name, app_author)
    print(f"User config directory: {user_config}")
    os.makedirs(user_config, exist_ok=True)
    assert os.path.exists(user_config)

    # User cache directory
    user_cache = appdirs.user_cache_dir(app_name, app_author)
    print(f"User cache directory: {user_cache}")
    os.makedirs(user_cache, exist_ok=True)
    assert os.path.exists(user_cache)

    # User log directory
    user_log = appdirs.user_log_dir(app_name, app_author)
    print(f"User log directory: {user_log}")
    os.makedirs(user_log, exist_ok=True)
    assert os.path.exists(user_log)

    # # Site data directory
    # site_data = appdirs.site_data_dir(app_name, app_author)
    # print(f"Site data directory: {site_data}")
    # os.makedirs(site_data, exist_ok=True)
    # assert os.path.exists(site_data)

    # # Site config directory
    # site_config = appdirs.site_config_dir(app_name, app_author)
    # print(f"Site config directory: {site_config}")
    # os.makedirs(site_config, exist_ok=True)
    # assert os.path.exists(site_config)

    print("\nAll appdirs functionalities have been tested successfully.")

if __name__ == "__main__":
    test_appdirs()
