# Integrates an IPython kernel with an asyncio event loop
# See: https://github.com/ipython/ipykernel/issues/21

have_ipython = False
try:
    get_ipython()
    have_ipython = True
except:
    pass

if have_ipython:
    import asyncio
    import itertools
    from ipykernel.eventloops import register_integration, enable_gui

    @register_integration('asyncio')
    def loop_asyncio(kernel):
        '''Start a kernel with asyncio event loop support.'''
        loop = asyncio.get_event_loop()

        def kernel_handler():
            loop.call_soon(kernel.do_one_iteration)
            loop.call_later(kernel._poll_interval, kernel_handler)

        loop.call_soon(kernel_handler)
        try:
            if not loop.is_running():
                loop.run_forever()
        finally:
            loop.close()

    print('[ipyloop] Started event loop')
    enable_gui('asyncio')
