# CM3604 Fact-checking coursework

## Setup
Python 3.10, then: pip install -r requirements.txt

## Run
python factchecking_main.py --mode word_overlap
python factchecking_main.py --mode entailment   (add --cuda only with a GPU)

## Team roles
- A: Part 1 (word overlap)
- B: Part 2 core (entailment model)
- C: Part 2 tuning (pruning, thresholds, logging)
- D: Part 3 (error analysis, report)
