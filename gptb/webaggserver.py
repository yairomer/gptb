import threading

import matplotlib.pyplot as plt
from matplotlib.backends.backend_webagg import WebAggApplication
from tornado.ioloop import IOLoop

def start_webagg_server():
    ## Set matplotlib's backend
    plt.switch_backend('webagg')

    ## Overwrite show method
    def blank_func(*args, **kwargs):
        pass
    plt.show = blank_func
    plt.ion()

    ## Start web server
    app = WebAggApplication()
    port = 9950
    while True:
        try:
            app.listen(port, address='0.0.0.0')
            break
        except:
            # print('!Port {} is not available'.format(port))
            port += 1
    print('Running web-backend on port: {}'.format(port))
    webagg_server_thread = threading.Thread(target=IOLoop.instance().start)
    webagg_server_thread.daemon = True
    webagg_server_thread.start()

    return webagg_server_thread
