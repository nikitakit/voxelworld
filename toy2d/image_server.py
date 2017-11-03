import tornado.ioloop
import tornado.web
import tornado.httpserver
import tornado
import logging
from tornado.options import define, options
import ssl
import os

define('port', default=9153, help='run on the given port', type=int)
define('key', default=os.path.join(os.path.dirname(__file__), "mykey.key"),
    help='path to SSL key file', type=str)
define('cert', default=os.path.join(os.path.dirname(__file__), "mycert.pem"),
    help='path to SSL cert file', type=str)

def make_app():
    return tornado.web.Application([
        (r"/scene_images/(.*)", tornado.web.StaticFileHandler, {'path': 'scene_images'}),
    ],
    )

if __name__ == "__main__":
    tornado.options.parse_command_line()

    app = make_app()

    if os.path.isfile(options.cert) and os.path.isfile(options.key):
        ssl_ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ssl_ctx.load_cert_chain(options.cert, options.key)
        logging.info("Using https")
    else:
        ssl_ctx = None
        logging.info("Not using https")

    http_server = tornado.httpserver.HTTPServer(app, ssl_options=ssl_ctx)
    http_server.listen(options.port)

    tornado.ioloop.IOLoop.current().start()
