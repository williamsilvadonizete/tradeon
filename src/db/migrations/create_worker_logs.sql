-- Create worker_logs table
CREATE TABLE IF NOT EXISTS worker_logs (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    etapa TEXT NOT NULL,
    payload JSONB NOT NULL,
    status TEXT NOT NULL,
    details TEXT
);

-- Create index on created_at for efficient TTL cleanup
CREATE INDEX IF NOT EXISTS idx_worker_logs_created_at ON worker_logs(created_at);

-- Create cleanup function
CREATE OR REPLACE FUNCTION cleanup_worker_logs()
RETURNS void AS $$
BEGIN
    DELETE FROM worker_logs 
    WHERE created_at < now() - interval '4 hours';
END;
$$ LANGUAGE plpgsql;

-- Create trigger for automatic cleanup
CREATE OR REPLACE FUNCTION trigger_cleanup_worker_logs()
RETURNS trigger AS $$
BEGIN
    PERFORM cleanup_worker_logs();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER cleanup_worker_logs_trigger
    AFTER INSERT ON worker_logs
    FOR EACH STATEMENT
    EXECUTE FUNCTION trigger_cleanup_worker_logs(); 