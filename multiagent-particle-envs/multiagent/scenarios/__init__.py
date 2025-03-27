import importlib.util
import os.path as osp

# Load a scenario from a path
def load(scenario_name):
    # Find the absolute path to the scenario file
    dirname = osp.dirname(__file__)
    path = osp.join(dirname, scenario_name)
    
    # Load module using importlib
    spec = importlib.util.spec_from_file_location("scenario", path)
    if spec is None:
        raise ImportError(f"Could not find scenario '{scenario_name}' at path: {path}")
        
    scenario_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scenario_module)
    
    return scenario_module