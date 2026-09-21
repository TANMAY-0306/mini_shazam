from db.database import init_db

def run():
    print("Applying schema to Neon DB...")
    init_db()
    print("Database tables ('songs', 'fingerprints') and indexes created successfully!")

if __name__ == "__main__":
    run()