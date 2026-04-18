-- PostgreSQL initialization script
-- Runs once on first container start (when data directory is empty)
-- Creates all databases and users needed by the ML platform

-- Airflow metadata database
CREATE DATABASE airflow;

-- MLflow experiment tracking database
CREATE DATABASE mlflow;

-- Crypto ETL data database with dedicated user
CREATE USER crypto WITH PASSWORD 'crypto123';
CREATE DATABASE crypto OWNER crypto;
GRANT ALL PRIVILEGES ON DATABASE crypto TO crypto;
