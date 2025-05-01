import os

venv_path = "/home/ccellerini/modular_model/.venv/lib/python3.10/site-packages"
output_file = "requirements.txt"

packages = []

for item in os.listdir(venv_path):
    # Check for package directories or .dist-info files
    if ".dist-info" in item:
        package_name = item.split("-")[0]
        version = item.split("-")[1] if len(item.split("-")) > 1 else ""
        if version:
            packages.append(f"{package_name}=={version}")

# Write to requirements.txt
with open(output_file, "w") as f:
    f.write("\n".join(packages))

print(f"Requirements written to {output_file}")
