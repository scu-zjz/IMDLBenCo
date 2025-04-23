
from IMDLBenCo.version import __version__, parse_version_info

current_version = '0.2.30'
future_version = '0.2.30'

print(current_version < future_version)


parsed_current = parse_version_info(current_version)
parsed_future = parse_version_info(future_version)
print(parsed_current)
print(parsed_future)
print(parsed_current < parsed_future)