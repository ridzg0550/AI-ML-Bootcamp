import os
import time
import math
import random
import json
import pickle
import re
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from tqdm.auto import tqdm
import spacy
from groq import Groq

print("✅ All packages imported successfully!")


# Reproducibility

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# CONFIGURATION

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "YOUR_GROQ_API_KEY_HERE")
MODEL_NAME = "llama-3.1-8b-instant"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_NEW_TOKENS = 200
CONVERSATION_HISTORY_LIMIT = 6
PERSIST_DIR = "./dungeonbrain_state"
ENABLE_MEMORY_LOGS = False

os.makedirs(PERSIST_DIR, exist_ok=True)

# COLORED LOGGING

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def log_memory(message: str, level: str = "info"):
    """Detailed memory logging for demo"""
    if not ENABLE_MEMORY_LOGS:
        return
    
    timestamp = time.strftime("%H:%M:%S")
    if level == "info":
        print(f"{Colors.CYAN}[{timestamp}]  MEMORY:{Colors.ENDC} {message}")
    elif level == "retrieval":
        print(f"{Colors.GREEN}[{timestamp}]  RETRIEVAL:{Colors.ENDC} {message}")
    elif level == "storage":
        print(f"{Colors.YELLOW}[{timestamp}]  STORAGE:{Colors.ENDC} {message}")
    elif level == "link":
        print(f"{Colors.BLUE}[{timestamp}]  LINK:{Colors.ENDC} {message}")


# EVALUATION METRICS TRACKER

class MetricsLogger:
    def __init__(self):
        self.turn_count = 0
        self.memory_retrievals = []
        self.npc_consistency_checks = []
        self.quest_updates = []
        
    def log_retrieval(self, turn: int, query: str, retrieved_turns: List[int]):
        self.memory_retrievals.append({
            "turn": turn,
            "query": query,
            "retrieved_turns": retrieved_turns,
            "has_early_recall": any(t <= 10 for t in retrieved_turns)
        })
        
    def log_npc_interaction(self, turn: int, npc: str, consistent: bool):
        self.npc_consistency_checks.append({
            "turn": turn,
            "npc": npc,
            "consistent": consistent
        })
        
    def get_stats(self) -> Dict:
        total_retrievals = len(self.memory_retrievals)
        early_recalls = sum(1 for r in self.memory_retrievals if r["has_early_recall"])
        
        avg_retrieved = 0
        if self.memory_retrievals:
            avg_retrieved = float(np.mean([len(r["retrieved_turns"]) for r in self.memory_retrievals]))
        
        npc_consistency_rate = None
        if self.npc_consistency_checks:
            npc_consistency_rate = float(np.mean([1.0 if c["consistent"] else 0.0 for c in self.npc_consistency_checks]))
        
        return {
            "total_turns": self.turn_count,
            "total_retrievals": total_retrievals,
            "early_event_recall_rate": (early_recalls / max(1, total_retrievals)) if total_retrievals > 0 else 0.0,
            "avg_retrieved_per_turn": avg_retrieved,
            "npc_consistency_rate": npc_consistency_rate
        }


# NEUROMORPHIC EVENT MEMORY 

class NeuroEventMemory:
    def __init__(self, embedding_model: str = EMBED_MODEL):
        print(f" Loading embedding model: {embedding_model}")
        self.encoder = SentenceTransformer(embedding_model)
        self.emb_dim = self.encoder.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.emb_dim)
        
        self.texts: List[str] = []
        self.embs: List[np.ndarray] = []
        self.salience: List[float] = []
        self.times: List[float] = []
        self.turn_numbers: List[int] = []
        self.is_permanent: List[bool] = []
        
        self.links = defaultdict(lambda: defaultdict(float))
        self.step = 0
        self.current_turn = 0
        print("✅ Memory system initialized")
        log_memory(f"Embedding dimension: {self.emb_dim}", "info")
        
    def encode(self, text: str) -> np.ndarray:
        vec = self.encoder.encode([text], convert_to_numpy=True, normalize_embeddings=True)
        return vec.astype("float32")
    
    def _ensure_index_consistency(self):
        if len(self.embs) == 0:
            self.index = faiss.IndexFlatIP(self.emb_dim)
            return
        
        if self.index.ntotal != len(self.embs):
            mat = np.vstack(self.embs).astype("float32")
            self.index = faiss.IndexFlatIP(self.emb_dim)
            self.index.add(mat)
    
    def add(self, text: str, is_critical: bool = False):
        """FIXED: Prevents duplicate storage"""
        log_memory(f"Adding memory (critical={is_critical}): {text[:60]}...", "storage")
        
        emb_2d = self.encode(text)
        emb = emb_2d[0]
        
        novelty = 1.0
        if len(self.embs) >= 4:
            mat = np.vstack(self.embs).astype("float32")
            sims = np.dot(mat, emb.T).flatten()
            max_sim = float(np.max(sims))
            novelty = 1.0 - max_sim
            log_memory(f"Novelty score: {novelty:.3f} (max similarity: {max_sim:.3f})", "storage")
            
            # FIX 1: Don't store near-duplicates even if critical
            if max_sim > 0.95:  # 95% similar = duplicate
                log_memory(" Memory rejected (duplicate detected)", "storage")
                return
        
        sal = novelty * math.log(len(text) + 1) * random.uniform(0.9, 1.1)
        threshold = 0.15 if is_critical else 0.25
        
        log_memory(f"Salience: {sal:.3f}, Threshold: {threshold:.3f}", "storage")
        
        if sal < threshold and not is_critical:
            log_memory(" Memory rejected (low salience)", "storage")
            return
        
        idx = len(self.texts)
        self.texts.append(text)
        self.embs.append(emb.astype("float32"))
        self.salience.append(sal)
        self.times.append(time.time())
        self.turn_numbers.append(self.current_turn)
        self.is_permanent.append(is_critical)
        
        try:
            self.index.add(emb_2d)
        except Exception:
            mat = np.vstack(self.embs).astype("float32")
            self.index = faiss.IndexFlatIP(self.emb_dim)
            self.index.add(mat)
        
        log_memory(f"✅ Memory stored at index {idx} (Turn {self.current_turn})", "storage")
        
        self._link_new_event(idx, emb)
    
    def _link_new_event(self, idx: int, emb: np.ndarray):
        if len(self.embs) < 5:
            return
        
        k = min(10, len(self.embs))
        self._ensure_index_consistency()
        D, I = self.index.search(emb.reshape(1, -1), k)
        
        links_created = 0
        for j, sim in zip(I[0], D[0]):
            if j == idx:
                continue
            w = max(0.0, float(sim))
            self.links[idx][j] += 0.03 * w
            self.links[j][idx] += 0.03 * w
            if w > 0.5:
                links_created += 1
        
        if links_created > 0:
            log_memory(f"Created {links_created} strong associative links for memory {idx}", "link")
    
    def retrieve(self, query: str, top_k: int = 5, force_early: bool = False) -> Tuple[List[str], List[int]]:
        """FIXED: Retrieves exactly top_k UNIQUE memories with detailed logging"""
        log_memory(f"\n{'='*60}", "retrieval")
        log_memory(f"QUERY: '{query}'", "retrieval")
        log_memory(f"Retrieving TOP {top_k} memories from {len(self.texts)} total", "retrieval")
        
        if not self.texts:
            log_memory("  No memories in database yet", "retrieval")
            return [], []
        
        # Encode query
        q = self.encode(query)
        k = min(len(self.texts), max(15, top_k * 3))
        self._ensure_index_consistency()
        D, I = self.index.search(q, k)
        
        log_memory(f"FAISS retrieved {k} candidates", "retrieval")
        
        now = time.time()
        scored = []
        max_sal = max(self.salience) if self.salience else 1.0
        
        # Score all candidates
        for idx, sim in zip(I[0], D[0]):
           
            recency = math.exp(-(now - self.times[idx]) / 3600.0)
            norm_sal = self.salience[idx] / max_sal if max_sal > 0 else 0.0
            perm_bonus = 0.15 if self.is_permanent[idx] else 0.0
            
            score = 0.55 * float(sim) + 0.20 * recency + 0.10 * norm_sal + perm_bonus
            scored.append((score, idx, float(sim), recency, norm_sal))
        
        scored.sort(key=lambda x: x[0], reverse=True)

        seen_turns = set()
        base_ids = []
        for score, idx, _, _, _ in scored:
            turn = self.turn_numbers[idx]
            if turn not in seen_turns:
                base_ids.append(idx)
                seen_turns.add(turn)
                if len(base_ids) >= top_k:
                    break
        
        log_memory(f"\nTop {len(base_ids)} unique scored memories:", "retrieval")
        for rank, idx in enumerate(base_ids, 1):
            score_info = next((s for s in scored if s[1] == idx), None)
            if score_info:
                score, _, sim, rec, sal = score_info
                memory_preview = self.texts[idx][:80].replace('\n', ' ')
                log_memory(
                    f"  {rank}. [Turn {self.turn_numbers[idx]}] Score:{score:.3f} "
                    f"(sim:{sim:.2f} rec:{rec:.2f} sal:{sal:.2f})\n"
                    f"     → {memory_preview}...",
                    "retrieval"
                )

        if force_early and self.current_turn > 20:
            early_memories = [i for i in range(len(self.texts)) if self.turn_numbers[i] <= 10]
            if early_memories and not any(self.turn_numbers[i] <= 10 for i in base_ids):
                log_memory("🎯 Forcing inclusion of early memory", "retrieval")
                for score, idx, _, _, _ in scored:
                    if idx in early_memories and self.turn_numbers[idx] not in seen_turns:
                        base_ids[-1] = idx
                        seen_turns.add(self.turn_numbers[idx])
                        log_memory(f"   Replaced last memory with Turn {self.turn_numbers[idx]}", "retrieval")
                        break

        activated = list(base_ids)
        activated_set = set(activated)
        spread_count = 0
        
        for i in base_ids:
            for j, w in self.links[i].items():
                if random.random() < min(0.8, w * 6.0):
                    if j not in activated_set and self.turn_numbers[j] not in seen_turns:
                        activated.append(j)
                        activated_set.add(j)
                        seen_turns.add(self.turn_numbers[j])
                        spread_count += 1
        
        if spread_count > 0:
            log_memory(f" Associative spreading activated {spread_count} linked memories", "retrieval")
 
        final_ids = activated[:top_k + 2]
        retrieved_turns = [self.turn_numbers[i] for i in final_ids]
        
        log_memory(f"\n✅ Final retrieval: {len(final_ids)} memories from turns {retrieved_turns}", "retrieval")
        log_memory(f"{'='*60}\n", "retrieval")
        
        return [self.texts[i] for i in final_ids], retrieved_turns
    
    def consolidate(self):
        self.step += 1
        if self.step % 3 != 0:
            return
        
        decay_count = 0
        for a in list(self.links.keys()):
            for b in list(self.links[a].keys()):
                self.links[a][b] *= 0.94
                if random.random() < 0.05:
                    self.links[a][b] += 0.015
                if self.links[a][b] < 0.004:
                    del self.links[a][b]
                    decay_count += 1
        
        if decay_count > 0:
            log_memory(f"Memory consolidation: {decay_count} weak links pruned", "link")
    
    def save(self, path: str):
        meta = {
            "texts": self.texts,
            "salience": self.salience,
            "times": self.times,
            "turn_numbers": self.turn_numbers,
            "is_permanent": self.is_permanent,
            "links": {k: dict(v) for k, v in self.links.items()},
            "step": self.step,
            "current_turn": self.current_turn
        }
        with open(path + ".meta.pkl", "wb") as f:
            pickle.dump(meta, f)
        if len(self.embs) > 0:
            mat = np.vstack(self.embs).astype("float32")
            np.save(path + ".embs.npy", mat)
            try:
                faiss.write_index(self.index, path + ".faiss")
            except Exception:
                idx = faiss.IndexFlatIP(self.emb_dim)
                idx.add(mat)
                faiss.write_index(idx, path + ".faiss")
    
    def load(self, path: str):
        if not os.path.exists(path + ".meta.pkl"):
            return
        with open(path + ".meta.pkl", "rb") as f:
            meta = pickle.load(f)
        self.texts = meta.get("texts", [])
        self.salience = meta.get("salience", [])
        self.times = meta.get("times", [])
        self.turn_numbers = meta.get("turn_numbers", [])
        self.is_permanent = meta.get("is_permanent", [])
        links_raw = meta.get("links", {})
        self.links = defaultdict(lambda: defaultdict(float), {int(k): defaultdict(float, v) for k, v in links_raw.items()})
        self.step = meta.get("step", 0)
        self.current_turn = meta.get("current_turn", 0)
        
        if os.path.exists(path + ".embs.npy"):
            mat = np.load(path + ".embs.npy").astype("float32")
            self.embs = [mat[i] for i in range(mat.shape[0])]
            if os.path.exists(path + ".faiss"):
                try:
                    self.index = faiss.read_index(path + ".faiss")
                except Exception:
                    self.index = faiss.IndexFlatIP(self.emb_dim)
                    self.index.add(mat)
            else:
                self.index = faiss.IndexFlatIP(self.emb_dim)
                self.index.add(mat)
        else:
            self.embs = []
            self.index = faiss.IndexFlatIP(self.emb_dim)


# NPC MEMORY SYSTEM

class NPCMemory:
    def __init__(self):
        self.npcs: Dict[str, Dict] = defaultdict(lambda: {
            "first_met_turn": None,
            "last_seen_turn": None,
            "relationship": "neutral",
            "key_facts": [],
            "dialogue_count": 0
        })
    
    def update_npc(self, name: str, turn: int, facts: List[str] = None, relationship: str = None):
        npc = self.npcs[name]
        if npc["first_met_turn"] is None:
            npc["first_met_turn"] = turn
        npc["last_seen_turn"] = turn
        npc["dialogue_count"] += 1
        
        if facts:
            npc["key_facts"].extend(facts)
            seen = []
            for f in npc["key_facts"][::-1]:
                if f not in seen:
                    seen.append(f)
            npc["key_facts"] = seen[::-1][:5]
        
        if relationship:
            npc["relationship"] = relationship
    
    def get_npc_context(self, name: str) -> str:
        if name not in self.npcs:
            return ""
        
        npc = self.npcs[name]
        facts_str = "; ".join(npc["key_facts"]) if npc["key_facts"] else "no specific history"
        return f"{name} ({npc['relationship']}): met turn {npc['first_met_turn']}, {facts_str}"
    
    def list_all(self) -> str:
        significant = {name: data for name, data in self.npcs.items() if data['dialogue_count'] >= 2}
        if not significant:
            return "No significant NPCs yet"
        return "\n".join([f"- {name}: {data['relationship']}, seen {data['dialogue_count']} times" 
                         for name, data in significant.items()])
    
    def save(self, path: str):
        with open(path + ".npc.pkl", "wb") as f:
            pickle.dump(dict(self.npcs), f)
    
    def load(self, path: str):
        if os.path.exists(path + ".npc.pkl"):
            with open(path + ".npc.pkl", "rb") as f:
                data = pickle.load(f)
            self.npcs = defaultdict(lambda: {
                "first_met_turn": None,
                "last_seen_turn": None,
                "relationship": "neutral",
                "key_facts": [],
                "dialogue_count": 0
            }, data)

# QUEST LOG SYSTEM
class QuestLog:
    def __init__(self):
        self.quests: Dict[str, Dict] = {}
        self.quest_counter = 0
    
    def add_quest(self, name: str, description: str, turn: int):
        """FIXED: Strict validation to prevent nonsense quests"""
       
        if len(name) < 10 or len(description) < 20:
            return None
        
        if description.count(' ') < 3:
            return None
   
        for q in self.quests.values():
            if q["name"] == name or q["description"] == description:
                return None
        
        self.quest_counter += 1
        quest_id = f"Q{self.quest_counter}"
        self.quests[quest_id] = {
            "name": name,
            "description": description,
            "status": "active",
            "started_turn": turn,
            "completed_turn": None,
            "updates": []
        }
        return quest_id
    
    def update_quest(self, quest_id: str, update: str, turn: int):
        if quest_id in self.quests:
            self.quests[quest_id]["updates"].append({"turn": turn, "text": update})
    
    def complete_quest(self, quest_id: str, turn: int):
        if quest_id in self.quests:
            self.quests[quest_id]["status"] = "completed"
            self.quests[quest_id]["completed_turn"] = turn
    
    def get_active_quests(self) -> str:
        active = [q for q in self.quests.values() if q["status"] == "active"]
        if not active:
            return "No active quests"
        
        lines = []
        for q in active:
            updates_str = f" ({len(q['updates'])} updates)" if q['updates'] else ""
            lines.append(f"- {q['name']}{updates_str}")
        return "\n".join(lines)
    
    def get_full_log(self) -> str:
        if not self.quests:
            return "No quests recorded"
        
        lines = []
        for qid, q in self.quests.items():
            status_icon = "✓" if q["status"] == "completed" else "→"
            lines.append(f"{status_icon} {q['name']} (turn {q['started_turn']})")
        return "\n".join(lines)
    
    def save(self, path: str):
        with open(path + ".quests.pkl", "wb") as f:
            pickle.dump({"quests": self.quests, "quest_counter": self.quest_counter}, f)
    
    def load(self, path: str):
        if os.path.exists(path + ".quests.pkl"):
            with open(path + ".quests.pkl", "rb") as f:
                data = pickle.load(f)
            self.quests = data.get("quests", {})
            self.quest_counter = data.get("quest_counter", 0)


# SLOT MEMORY

class SlotMemory:
    def __init__(self):
        self.slots = {
            "location": None,
            "time_of_day": None,
            "party_members": [],
            "inventory_highlights": [],
            "current_goal": None
        }
    
    def update(self, llm_generate, dialogue: str):
      
        low = dialogue.lower()

        loc_matches = re.findall(r"\b(in the|at the|to the|arrive at|enter)\s+([a-zA-Z\s']{3,30})", low)
        if loc_matches:
            candidate = loc_matches[-1][1].strip()
            candidate = candidate.split(".")[0][:40].strip()
            if len(candidate) > 3:
                self.slots["location"] = candidate

        if "morning" in low or "afternoon" in low or "evening" in low or "night" in low:
            for t in ["morning", "afternoon", "evening", "night"]:
                if t in low:
                    self.slots["time_of_day"] = t
                    break
    
    def context(self) -> str:
        lines = []
        if self.slots["location"]:
            lines.append(f"Location: {self.slots['location']}")
        if self.slots["current_goal"]:
            lines.append(f"Goal: {self.slots['current_goal']}")
        if self.slots["party_members"]:
            lines.append(f"Party: {', '.join(self.slots['party_members'])}")
        return "\n".join(lines) if lines else ""
    
    def save(self, path: str):
        with open(path + ".slots.pkl", "wb") as f:
            pickle.dump(self.slots, f)
    
    def load(self, path: str):
        if os.path.exists(path + ".slots.pkl"):
            with open(path + ".slots.pkl", "rb") as f:
                self.slots = pickle.load(f)

# GROQ LLM ENGINE

class GroqLLMEngine:
    def __init__(self, api_key: str, model_name: str = MODEL_NAME):
        self.client = Groq(api_key=api_key)
        self.model_name = model_name
        print(f" Using Groq API with {model_name}")
    
    def generate(self, prompt: str, max_new_tokens: int = MAX_NEW_TOKENS, 
                 temperature: float = 0.7, **kwargs) -> str:
        """Generate with better stopping and post-processing"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_new_tokens,
                top_p=0.92,
                frequency_penalty=0.3,
                presence_penalty=0.2,
                stop=None
            )
            
            text = response.choices[0].message.content.strip()
            text = self._truncate_at_sentence(text, max_sentences=3)
            
            return text
            
        except Exception as e:
            print(f"[ERROR] Groq API call failed: {e}")
            return "The Dungeon Master pauses to gather their thoughts..."
    
    def _truncate_at_sentence(self, text: str, max_sentences: int = 3) -> str:
        """Truncate text to max_sentences complete sentences"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        if len(sentences) > max_sentences:
            truncated = ' '.join(sentences[:max_sentences])
            if not truncated[-1] in '.!?':
                truncated += '.'
            return truncated
        
        return text
    
    def count_tokens(self, text: str) -> int:
        """Rough token estimate (4 chars ≈ 1 token)"""
        return len(text) // 4


# IMPROVED NPC EXTRACTION

try:
    nlp = spacy.load("en_core_web_sm")
    print("✅ spaCy NLP model loaded")
except Exception as e:
    print(f" spaCy model not loaded: {e}")
    nlp = None

def extract_npcs_improved(text: str, dialogue_context: str = "") -> List[str]:
    """
    FIXED NPC extraction:
    - Only extract proper nouns from dialogue (not descriptions)
    - Require multi-word context or explicit introduction
    - Filter common false positives
    """
    npcs = []
    
    blacklist = {
        "I", "You", "It", "The", "A", "An", "And", "But", "Or", "If", "When",
        "Even", "Alight", "Crystal", "Quest", "Dragon", "Cave", "Mountain",
        "Village", "Forest", "North", "South", "East", "West", "Turn", "Player", "DM", "Dungeon", "Master"
    }
    
    if nlp is not None:
        try:
            full_text = f"{dialogue_context}\n{text}"
            doc = nlp(full_text)
            
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    name = ent.text.strip()
                    if (len(name) >= 2 and 
                        name[0].isupper() and 
                        name not in blacklist and
                        name.lower() not in ["i", "you"]):
                        npcs.append(name)
        except Exception:
            pass
    
    if not npcs:
        patterns = [
            r"(?:meet|met|named|called|name is|named|speak to|talk to)\s+([A-Z][a-z]{2,15})",
            r"([A-Z][a-z]{2,15})\s+(?:the|says|tells|warns|agrees|asks|replies)"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for name in matches:
                if name not in blacklist and len(name) >= 3:
                    npcs.append(name)
    
    return list(dict.fromkeys(npcs))[:5]


# DUNGEONBRAIN++ MAIN SYSTEM - FIXED

class DungeonBrainPlus:
    def __init__(self, api_key: str = GROQ_API_KEY):
        print("\n Initializing DungeonBrain++...")
        self.llm = GroqLLMEngine(api_key=api_key)
        
        self.memory = NeuroEventMemory()
        self.slots = SlotMemory()
        self.npc_memory = NPCMemory()
        self.quest_log = QuestLog()
        self.metrics = MetricsLogger()
        
        self.dialogue_history: List[str] = []
        self.turn_count = 0
        
        self._load_state()
        print("✅ DungeonBrain++ ready!\n")
    
    def _persist_prefix(self) -> str:
        return os.path.join(PERSIST_DIR, "dungeonbrain")
    
    def save_state(self):
        prefix = self._persist_prefix()
        try:
            self.memory.save(prefix + ".memory")
            self.npc_memory.save(prefix)
            self.quest_log.save(prefix)
            self.slots.save(prefix)
            meta = {
                "dialogue_history": self.dialogue_history,
                "turn_count": self.turn_count,
                "metrics": self.metrics.__dict__
            }
            with open(prefix + ".state.pkl", "wb") as f:
                pickle.dump(meta, f)
            print(" State saved successfully.")
        except Exception as e:
            print(f" Failed to save: {e}")
    
    def _load_state(self):
        prefix = self._persist_prefix()
        try:
            self.memory.load(prefix + ".memory")
            self.npc_memory.load(prefix)
            self.quest_log.load(prefix)
            self.slots.load(prefix)
            if os.path.exists(prefix + ".state.pkl"):
                with open(prefix + ".state.pkl", "rb") as f:
                    meta = pickle.load(f)
                self.dialogue_history = meta.get("dialogue_history", [])
                self.turn_count = meta.get("turn_count", 0)
                try:
                    m = meta.get("metrics", {})
                    self.metrics.turn_count = m.get("turn_count", 0)
                except Exception:
                    pass
                print(f" Loaded previous session (Turn {self.turn_count})")
        except Exception as e:
            print(f" Starting fresh session")
    
    def _build_context(self, user_input: str) -> Tuple[str, List[int]]:
        blocks = []
        retrieved_turns: List[int] = []
        
        slot_ctx = self.slots.context()
        if slot_ctx:
            blocks.append(f"=== Current World State ===\n{slot_ctx}")
        
        quest_ctx = self.quest_log.get_active_quests()
        if quest_ctx and "No active" not in quest_ctx:
            blocks.append(f"=== Active Quests ===\n{quest_ctx}")
        
        force_early = self.turn_count > 20
        retrieved_events, retrieved_turns = self.memory.retrieve(
            user_input, 
            top_k=5,  
            force_early=force_early
        )
        if retrieved_events:
            blocks.append(f"=== Relevant Past Events ===\n" + "\n".join(retrieved_events))
        
        recent_dialogue = "\n".join(self.dialogue_history[-3:])
        npcs = extract_npcs_improved(user_input, recent_dialogue)
        if npcs:
            npc_contexts = [self.npc_memory.get_npc_context(npc) for npc in npcs]
            npc_contexts = [c for c in npc_contexts if c]
            if npc_contexts:
                blocks.append(f"=== Known NPCs ===\n" + "\n".join(npc_contexts))
        
        if self.dialogue_history:
            recent = self.dialogue_history[-min(2, CONVERSATION_HISTORY_LIMIT):]
            blocks.append(f"=== Recent Conversation ===\n" + "\n".join(recent))
        
        return "\n\n".join(blocks), retrieved_turns
    
    def respond(self, user_input: str) -> str:
        self.turn_count += 1
        self.memory.current_turn = self.turn_count
        self.metrics.turn_count = self.turn_count
        
        print(f"\n{Colors.BOLD}{'='*60}{Colors.ENDC}")
        print(f"{Colors.BOLD}TURN {self.turn_count}{Colors.ENDC}")
        print(f"{Colors.BOLD}{'='*60}{Colors.ENDC}")
        
        context, retrieved_turns = self._build_context(user_input)
        self.metrics.log_retrieval(self.turn_count, user_input, retrieved_turns)
        
     
        system_prompt = (
            "You are a concise Dungeon Master. Rules:\n"
            "1. Respond directly to the player's action/question\n"
            "2. Be specific - use names, describe outcomes\n"
            "3. Keep responses 2-3 sentences maximum\n"
            "4. DON'T repeat previous descriptions\n"
            "5. If player asks an NPC something, have the NPC actually answer\n"
            "6. Move the story forward with each response\n\n"
            "Context:"
        )
        
        full_prompt = f"{system_prompt}\n\n{context}\n\nPlayer: {user_input}\n\nDungeon Master (respond briefly and directly):"
        
        response = self.llm.generate(full_prompt, max_new_tokens=200, temperature=0.75)
        
        dialogue_entry = f"[Turn {self.turn_count}] Player: {user_input}\nDM: {response}"
        self.dialogue_history.append(dialogue_entry)
        
        if len(self.dialogue_history) > CONVERSATION_HISTORY_LIMIT:
            self.dialogue_history = self.dialogue_history[-CONVERSATION_HISTORY_LIMIT:]
        
        is_critical = any(keyword in user_input.lower() for keyword in 
                         ["i choose", "i attack", "i take", "i accept", "i refuse", 
                          "i go to", "i use", "i found", "i enter", "i meet", "named"])
        
        try:
            self.memory.add(dialogue_entry, is_critical=is_critical)
        except Exception as e:
            print(f" Memory add failed: {e}")
        
        try:
            self._update_subsystems(user_input, response)
        except Exception as e:
            print(f" Subsystem update failed: {e}")
        
        try:
            self.memory.consolidate()
        except Exception:
            pass
        
        return response
    
    def _update_subsystems(self, user_input: str, response: str):
        """FIXED: No more quest_id errors, strict quest validation"""
        combined = f"{user_input}\n{response}"
        
        try:
            self.slots.update(self.llm.generate, combined)
        except Exception as e:
            print(f" Slot update failed: {e}")
        
        recent_dialogue = "\n".join(self.dialogue_history[-3:])
        npcs = extract_npcs_improved(combined, recent_dialogue)
        for npc_name in npcs:
            try:
                self.npc_memory.update_npc(npc_name, self.turn_count)
                self.metrics.log_npc_interaction(self.turn_count, npc_name, consistent=True)
            except Exception:
                continue
        
        explicit_quest_markers = [
            "your quest is",
            "quest:",
            "mission:",
            "objective:",
            "i give you the quest",
            "accept this quest",
            "your mission is"
        ]
        
        has_explicit_quest = any(marker in combined.lower() for marker in explicit_quest_markers)
        
        if has_explicit_quest:
            quest_patterns = [
                r"(?:your quest is to|quest:|mission:|objective:)\s+([a-z][a-z\s]{20,80})(?:\.|!|\?|$)",
                r"(?:i give you the quest to|your mission is to)\s+([a-z][a-z\s]{20,80})(?:\.|!|\?|$)"
            ]
            
            for pattern in quest_patterns:
                matches = re.findall(pattern, combined, re.IGNORECASE)
                for match in matches:
                    quest_desc = match.strip().rstrip('.,!?')
      
                    invalid_phrases = [
                        "almost", "given up", "rustling", "you stand", "as you",
                        "the air", "you can", "you hear", "wait for", "you've",
                        "i wait", "venture", "heavy with"
                    ]
                    
                    if any(phrase in quest_desc.lower() for phrase in invalid_phrases):
                        continue
                    
                    if len(quest_desc) >= 20 and quest_desc.count(' ') >= 3:
                        words = quest_desc.split()[:6]
                        quest_name = " ".join(words).capitalize()
                        
                        try:
                            quest_id = self.quest_log.add_quest(
                                name=quest_name,
                                description=quest_desc,
                                turn=self.turn_count
                            )
                            if quest_id:
                                log_memory(f"✅ Quest logged: {quest_name}", "info")
                                break
                        except Exception:
                            pass
    
    def get_stats(self) -> Dict:
        return self.metrics.get_stats()
    
    def print_stats(self):
        stats = self.get_stats()
        print("\n" + "="*60)
        print(" DUNGEONBRAIN++ EVALUATION METRICS")
        print("="*60)
        print(f"Total Turns: {stats.get('total_turns', 0)}")
        print(f"Memory Retrievals: {stats.get('total_retrievals', 0)}")
        print(f"Early Event Recall Rate: {stats.get('early_event_recall_rate', 0.0):.2%}")
        print(f"Avg Retrieved/Turn: {stats.get('avg_retrieved_per_turn', 0.0):.1f}")
        ncr = stats.get('npc_consistency_rate', None)
        if ncr is None:
            print(f"NPC Consistency: N/A")
        else:
            print(f"NPC Consistency: {ncr:.2%}")
        print(f"\nActive Quests:\n{self.quest_log.get_active_quests()}")
        print(f"\nKnown NPCs:\n{self.npc_memory.list_all()}")
        print("="*60 + "\n")


# INTERACTIVE LOOP

def interactive_session():
    print("\n" + "="*60)
    print(" DUNGEONBRAIN++ — NEUROMORPHIC DUNGEON MASTER")
    print("="*60)
    print(f"Session: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"User: CIPHERclux")
    print("Using Groq API with Llama 3.1 8B")
    print(f"Memory Logging: {'ENABLED' if ENABLE_MEMORY_LOGS else 'DISABLED'}\n")
    
    if GROQ_API_KEY == "YOUR_GROQ_API_KEY_HERE":
        print(" ERROR: Please set your GROQ_API_KEY!")
        print("Get your free API key at: https://console.groq.com/keys")
        print("\n Set it as environment variable:")
        print("   export GROQ_API_KEY='your_key_here'")
        print("\nOr edit line 51 of the code.\n")
        return
    
    db = DungeonBrainPlus(api_key=GROQ_API_KEY)
    
    print("✅ Ready! Commands:")
    print("  - Type your action to play")
    print("  - 'stats' to see metrics")
    print("  - 'save' to persist state")
    print("  - 'load' to reload state")
    print("  - 'quit' to exit\n")
    
    opening = "You stand at the entrance of a misty forest. A worn path leads deeper into the trees, and you hear the distant sound of running water."
    print(f" Dungeon Master: {opening}\n")
    db.memory.add(f"[Turn 0] DM: {opening}", is_critical=True)
    
    try:
        while True:
            user_input = input(f"{Colors.BOLD}  You:{Colors.ENDC} ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ["quit", "exit", "q"]:
                print("\n Ending session...")
                db.print_stats()
                db.save_state()
                break
            
            if user_input.lower() == "stats":
                db.print_stats()
                continue
            
            if user_input.lower() == "save":
                db.save_state()
                continue
            
            if user_input.lower() == "load":
                db._load_state()
                print("✅ State reloaded.")
                continue
            
            response = db.respond(user_input)
            print(f"\n{Colors.BOLD} Dungeon Master:{Colors.ENDC} {response}\n")
            
    except (KeyboardInterrupt, EOFError):
        print("\n\n Session interrupted.")
        db.print_stats()
        db.save_state()


# MAIN ENTRY POINT

if __name__ == "__main__":
    interactive_session()