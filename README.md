# Wikontext
## Smarter page previews for a smoother Wikipedia experience.

Show the most relevant parts of a linked Wikipedia article with the hover of your mouse

### Usage
```
git clone 
cd wikontext
mkdir models
./download_models.sh models
DOCKER_BUILDKIT=1 docker build -t wikontext .
docker run -it -d -p 80:80 -p 443:443 \
    -v $(pwd)/letsencrypt:/etc/letsencrypt \
    -v $(pwd)/nginx/sites-available:/etc/nginx/sites-available \
    -v $(pwd)/nginx/sites-enabled:/etc/nginx/sites-enabled \
    -v $(pwd)/models:/models \
    wikontext
```
