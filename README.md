# Wikontext: Smarter page previews for a smoother Wikipedia experience.

Show the most relevant parts of a linked Wikipedia article with the hover of your mouse

## Usage

### Package Chrome Extension
```
git clone https://github.com/knowbodynos/wikontext.git
cd wikontext/chrome_extension
npm install
npm run package
```

### Serve Backend
```
git clone https://github.com/knowbodynos/wikontext.git
cd wikontext
./setup_env.sh
./download_models.sh models
CERT_EMAIL=<your-email> ./docker.sh
```
