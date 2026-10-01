"""One module per method; each scores s(A→B), how strongly having skill A implies having skill B."""
from .box import BoxEmbedding
from .cosine import cosine, cosine_generality
from .cross_encoder import CrossEncoder
from .distill import distill, folded_vectors
from .dual import DualEncoder
from .hybrid import hybrid
from .hyperbolic import EuclideanEmbedding, PoincareEmbedding
from .order import OrderEmbedding
from .pair_mlp import PairMLP
from .transe import TransE
