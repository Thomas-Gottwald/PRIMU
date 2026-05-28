from .rand_selector import RandSelector
from .PRIMU_maps_selector import PRIMUMapsSelector

methods_dict = {"rand": RandSelector,
                "PRIMU": PRIMUMapsSelector}
