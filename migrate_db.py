import psycopg2
import yaml
import os

def migrate():
    config_path = 'config.yaml'
    if not os.path.exists(config_path):
        print("Config file not found")
        return

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    db_conf = config.get('database', {})
    if db_conf.get('engine') != 'postgresql':
        print("Not using postgresql, skipping migration")
        return

    try:
        conn = psycopg2.connect(
            host=db_conf.get('host', 'localhost'),
            port=db_conf.get('port', 5432),
            user=db_conf.get('user', 'postgres'),
            password=db_conf.get('password', '123'),
            dbname=db_conf.get('name', 'autocr')
        )
        cur = conn.cursor()
        cur.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS error_message TEXT")
        conn.commit()
        print("Migration successful: error_message column added to documents table.")
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Migration failed: {e}")

if __name__ == "__main__":
    migrate()
