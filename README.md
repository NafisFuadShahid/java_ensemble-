# Install dependencies
pip install -r requirements.txt

# Full ensemble training (~2-3 hours on T4)
python java_ensemble_training.py

# OR quick training (~1 hour)
python quick_start_training.py

# Generate test predictions after training
python inference.py --test_path /path/to/java_test.parquet --output submission.csv
