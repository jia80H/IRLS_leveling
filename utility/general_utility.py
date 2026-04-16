# def a function: if folder is not exist, create it
def ge_data_path(path):
    """
    Creates a new directory at the specified path if it doesn't already exist.

    Args:
        path (str): The path of the directory to create.

    Returns:
        path
    """
    import os
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
    return path
