"""Fine-tuned cross-encoder: one transformer pass reads (A text, B text) → logit of P(B | A).
Order-sensitive input, so asymmetric. Too slow to score a whole catalogue per query: rerank a shortlist."""
import torch
import torch.nn.functional as F

from ..data import DEVICE


class CrossEncoder:
    def __init__(self, texts, name="BAAI/bge-reranker-base", max_len=96):
        """texts: the text of every node, by index."""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.texts, self.max_len = texts, max_len
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).to(DEVICE)
        self.opt = None

    def score_texts(self, texts_a, texts_b):
        enc = self.tokenizer(texts_a, texts_b, truncation=True, max_length=self.max_len, padding=True, return_tensors="pt")
        return self.model(**enc.to(DEVICE)).logits.squeeze(-1)

    def logits(self, a, b):
        return self.score_texts([self.texts[i] for i in a], [self.texts[j] for j in b])

    def pairs(self, a, b, batch=256):
        """Aligned pairs, inference. fp16 on MPS: 2.5× faster, logits within 0.06 of fp32."""
        self.model.eval()
        with torch.no_grad(), torch.autocast(DEVICE, dtype=torch.float16, enabled=DEVICE == "mps"):
            return torch.cat([self.logits(a[i:i + batch].tolist(), b[i:i + batch].tolist()).float() for i in range(0, len(a), batch)])

    def __call__(self, a, b):
        return self.pairs(a.repeat_interleave(len(b)), b.repeat(len(a))).view(len(a), len(b))

    def fit_epoch(self, labelled, lr=2e-5, batch=32):
        """labelled: [(A id, B id, y)], e.g. training.labelled_pairs(graph)."""
        if self.opt is None: self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.model.train()
        for i in range(0, len(labelled), batch):
            a, b, y = zip(*labelled[i:i + batch])
            loss = F.binary_cross_entropy_with_logits(self.logits(a, b), torch.tensor(y, device=DEVICE))
            self.opt.zero_grad(); loss.backward(); self.opt.step()
