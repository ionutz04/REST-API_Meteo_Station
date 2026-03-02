-- Allow admin user to connect from any host
CREATE USER IF NOT EXISTS 'admin'@'%' IDENTIFIED BY 'ionutqwerty';
GRANT ALL PRIVILEGES ON PRODUCERS.* TO 'admin'@'%';
FLUSH PRIVILEGES;
