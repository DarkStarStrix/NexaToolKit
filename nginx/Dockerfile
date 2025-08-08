# Use the official Nginx image
FROM nginx:alpine

# Remove default configuration
RUN rm /etc/nginx/conf.d/default.conf

# Copy our custom Nginx configuration
COPY nginx.conf /etc/nginx/conf.d/

# Create a directory for self-signed certificates
RUN mkdir -p /etc/nginx/certs

# Generate self-signed certificate for local HTTPS
RUN openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout /etc/nginx/certs/selfsigned.key \
    -out /etc/nginx/certs/selfsigned.crt \
    -subj "/C=US/ST=State/L=City/O=Development/CN=localhost"

# Expose ports
EXPOSE 80
EXPOSE 443

