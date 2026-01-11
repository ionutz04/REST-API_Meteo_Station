#!/bin/bash

# Script to generate self-signed SSL certificates for the Flask server
# Run this script once before starting the server

# Create certs directory if it doesn't exist
mkdir -p certs
# Generate private key
openssl genrsa -out certs/server.key 2048

# Generate self-signed certificate (valid for 365 days)
# You can modify the subject (-subj) with your own details
openssl req -new -x509 \
    -key certs/server.key \
    -out certs/server.crt \
    -days 365 \
    -subj "/C=US/ST=State/L=City/O=Organization/OU=Unit/CN=localhost"

# Set appropriate permissions
chmod 600 certs/server.key
chmod 644 certs/server.crt
