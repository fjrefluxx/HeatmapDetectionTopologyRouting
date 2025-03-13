# Topology-Aware Routing

Python implementation of a grid-based hotspot detection and routing algorithm, 
based on an original Java implementation (cf. [the paper](#acknowledging-this-work)).

The idea is to input location measurements of devices from a wide-spread distributed wireless network, assuming a clustered network topology.
The approach allows automatic detection of clusters and TSP-like route calculation for a traversing devices, such as a UAV. 
In case of movement between network clusters, routing tries to increase coverage of moving nodes between clusters.

### Features
- Heatmap creation and population with measurements (locations)
- Using simpy, the heatmap's measurement counters can be degraded over time. Requires manual start. 
- Local hotspot detection. Gaussian smoothing is used to address cluster spread over multiple cells, a highpass filter is required to filter non-zero low-density heatmap cells. 
- A hotspot tour is calculated using the Christofides algorithm to solve TSP
- Individual route calculation between consecutive hotspots in the tour incorporates long-term measurements to increase coverage of moving network nodes between hotspots.

### Requirements
Tested with Python 3.12
- simpy >= 4.0 
- networkx >= 3.4
- numpy >= 2.2

## Example
```python
import numpy as np

from util.Heatmap import Heatmap
from TopologyAwareRouting import TopologyAwareRouting

# create size=1000x1000 heatmap, 20x20 cells (size=50x50), sigma of 50 for kernel size
heatmap = Heatmap((0, 0), (1000, 1000), 50, 50)

# distribute 1000 measurements around location 100/100
for i in range(1000):
    heatmap.input_measurement((np.random.randint(75, 125), np.random.randint(75, 125)))

# distribute 500 measurements around location 500/900
for i in range(500):
    heatmap.input_measurement((np.random.randint(475, 525), np.random.randint(875, 925)))

# distribute 1500 measurements around location 800/200
for i in range(1500):
    heatmap.input_measurement((np.random.randint(750, 850), np.random.randint(150, 250)))

# calculate path, with a starting location at 500/500
path = TopologyAwareRouting.waypoints(heatmap, (500, 500))

print(path)
```



## Acknowledging this work

If you use this software in a scientific publication, please cite the following paper:

```BibTeX
@inproceedings{zobel2020topology,
  title={Topology-Aware Path Planning for In-Transit Coverage of Aerial Post-Disaster Communication Assistance Systems},
  author={Zobel, Julian and Becker, Benjamin and Kundel, Ralf and Lieser, Patrick and Steinmetz, Ralf},
  booktitle={2020 IEEE 45th LCN Symposium on Emerging Topics in Networking (LCN Symposium)},
  pages={88--98},
  year={2020},
  organization={IEEE}
}
```

## License
Licensed under <a href="LICENSE">GPL-3.0</a>.