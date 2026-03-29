# Use official Nginx image
FROM nginx:alpine

# Set working directory (optional)
WORKDIR /usr/share/nginx/html

# Copy your static files (index.html, css, js, etc.)
COPY index.html ./
# If you have a folder called "assets" or "css", copy it too:
# COPY assets ./assets
# COPY css ./css

# Expose default HTTP port
EXPOSE 80

# Start Nginx (already the default CMD in the image)
# CMD ["nginx", "-g", "daemon off;"]