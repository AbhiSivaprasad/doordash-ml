import re


def get_latest_version_number(api, artifact_name: str) -> int:
    """
    Given the name of a W&B artifact, return the latest version number.
    If the artifact does not exist then return None
    """
    try:
        artifact = api.artifact(f"{artifact_name}:latest")
    except:
        # artifact does not exist
        # TODO: make error catch more specific
        return None

    version_alias = [alias for alias in artifact.aliases if re.search("^v[0-9]+$", alias)]

    if len(version_alias) == 0:
        raise ValueError(f"Artifact {artifact_name} does not have a version alias")

    if len(version_alias) > 1:
        raise ValueError("Latest artifact has multiple version aliases:", version_alias)

    version_alias = version_alias[0]
    version_number = int(re.search("[0-9]+$", version_alias).group())
    return version_number
