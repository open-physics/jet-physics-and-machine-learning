## Running Docker

The following set-up is scripted as well in a `script.py` file.

Assume IMAGE_NAME as `jet-physics`.

### To build Docker
`docker build -t jet-physics .`

Or,

`python script.py --build`

### To enter bash shell of Docker
`docker run -it jet-physics bash`

Or,

`python script.py --run`
