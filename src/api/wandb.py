import re


def get_latest_artifact_identifier(api, artifact_identifier: str) -> str:
    """
    Given the identifier of a W&B artifact, return the latest artifact name.
    Note, a specific version alias must be given e.g. "latest"
        e.g. "artifact_name:latest" --> "artifact_name:v5"

    If the artifact does not exist then return None
    """
    try:
        # latest is default if not included
        artifact = api.artifact(artifact_identifier)
    except:
        # artifact does not exist
        # TODO: make error catch more specific
        return None
    
    return artifact.name 
