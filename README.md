# 🧠 Cognitive Commuter Graph: Teaching AI to Think Like a City Walker 🚶‍♂️

**Author:** Apoorv Thite  
**Tech Stack:** PyTorch Geometric · GeoPandas · OSMnx · NetworkX · Folium · Matplotlib · NumPy · Scikit-learn  
**Notebook:** [`Cognitive_Commuter_Graph.ipynb`](./Cognitive_Commuter_Graph.ipynb)

---

## 🌍 Overview
Cities are full of micro-decisions — every turn, shortcut, and detour reflects how humans weigh **comfort, time, and effort**.  
This project explores whether **AI can understand that**.

The **Cognitive Commuter Graph** models **human-like pedestrian routing** by combining **Graph Neural Networks (GNNs)** with **urban network data**.  
Using **City2Graph** and **OSMnx**, Manhattan’s road network was transformed into a graph, where each edge represents a street segment and each node a real-world intersection.  

Unlike navigation tools such as **Google Maps**, which optimize for fixed metrics (e.g., shortest time or distance), this model learns **how people actually move** — preferring smoother, more consistent paths that reflect cognitive trade-offs in real-world walking behavior.

---

## 🧩 Objectives
- Build a **GNN-based behavioral model** that predicts the next likely street segment a pedestrian takes.
- Use **contrastive learning with hard negatives** to distinguish between plausible and implausible next moves.
- Incorporate **geospatial and physical features** — length (m), time (s), and slope (%) — to represent environmental constraints.
- Visualize **decision entropy**, **learned edge costs**, and **saliency maps** to interpret model behavior.
- Compare learned routing patterns with traditional shortest-path algorithms (Dijkstra/A*).

---

## 🗺️ Dataset & Preprocessing
- **City:** Manhattan, New York City 🏙️  
  Chosen for its high walkability and structured grid layout, ideal for modeling pedestrian cognition.
- **Data Source:** [OpenStreetMap via OSMnx](https://github.com/gboeing/osmnx)
- **Graph Construction:**
  - Combined drive and walk networks using `network_type="drive"` and `"walk"`.
  - Computed edge attributes:
    - `length_m`: street segment length in meters  
    - `time_s`: walking time (length / 1.3 m/s)  
    - `slope_pct`: slope percentage from node elevation difference  

---

## 🧠 Model Architecture
### Graph Neural Network (GNN)
- **Framework:** PyTorch Geometric  
- **Encoder:** 2× GraphSAGE convolution layers → node embeddings  
- **Edge MLP Head:**  
  - Inputs: concatenated (u, v) node embeddings + edge features  
  - Layers: `Linear → BatchNorm → ReLU → Linear → Sigmoid`
  - Output: learned edge “comfort cost”  
- **Training Strategy:**
  - **Contrastive loss with hard negatives** (K=8)
  - **AdamW optimizer** with learning rate scheduling  
  - **Frozen encoder** phase + **fine-tuning** of the second GNN layer

### Learned-Cost A* Routing
- Replaces static weights with dynamic, **GNN-learned edge costs**.
- Generates **context-aware paths** that mimic real human navigation — balancing comfort and efficiency.

---

## 📊 Key Visualizations
| Visualization | Description |
|----------------|--------------|
| **Heatmap of Learned Edge Costs** | Displays how the model values different streets — lighter = easier, darker = costlier. |
| **Entropy Map** | Highlights intersections where the model is least certain — real-world “hesitation zones.” |
| **Route Comparison (Learned vs Time-shortest)** | Overlays two routes to visualize behavioral differences. |
| **Slope Profile Analysis** | Shows elevation change across routes — learned paths tend to avoid sharp climbs. |
| **Edge Feature Saliency (Grad-based)** | Quantifies influence of length, time, and slope in decision-making. |

---

## 📈 Results & Insights
- **Top-1 Accuracy:** ~37% (≈4× random baseline)  
- **Decision Entropy:** Identified key intersections where pedestrians hesitate most.  
- **Gradient Saliency:** Time and distance dominate routing decisions, slope has lesser but consistent influence.  
- **Behavioral Insight:** Learned routes are often slightly longer but more consistent, smoother, and cognitively intuitive.

---

## 🧮 Techniques Used
- **Graph Neural Networks (GNNs)**  
- **Contrastive Learning (Hard Negatives)**  
- **Learned-Cost A* Search**  
- **Entropy & Saliency Analysis**  
- **Feature Normalization (z-score)**  
- **OSMnx Geospatial Graphs & GeoPandas Manipulation**

---

## 🧰 Tools & Libraries
| Category | Libraries |
|-----------|------------|
| **Deep Learning** | PyTorch, PyTorch Geometric |
| **Graph Processing** | NetworkX, OSMnx |
| **Geospatial** | GeoPandas, Folium, Shapely |
| **Visualization** | Matplotlib, Folium, Branca |
| **Data Handling** | NumPy, Pandas |

---

## 🚀 How to Run
1. **Clone this repository:**
   ```bash
   git clone https://github.com/<your-username>/Cognitive-Commuter-Graph.git
   cd Cognitive-Commuter-Graph
