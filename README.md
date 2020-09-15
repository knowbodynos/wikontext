# Wikontext
## Smarter page previews for a smoother Wikipedia experience.

Show the most relevant parts of a linked Wikipedia article with the hover of your mouse

### Usage
```
git clone 
cd wikontext
mkdir models
./download_models.sh models
docker build -t wikontext .
docker run -it -d -v $(pwd)/models:/models -p 8000:80 wikontext
```
