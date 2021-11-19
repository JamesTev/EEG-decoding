import gc
import micropython


def print_memory_diagnostics():
    # see https://docs.micropython.org/en/latest/reference/constrained.html
    gc.enable()

    gc.collect()
    micropython.mem_info()
    print("-----------------------------")
    print("Initial free: {} allocated: {}".format(gc.mem_free(), gc.mem_alloc()))

    def func():
        # dummy memory assignment
        import urandom

        x = [urandom.random() for i in range(100)]
        return

    gc.collect()
    print("Func definition: {} allocated: {}".format(gc.mem_free(), gc.mem_alloc()))
    func()
    print("Func run free: {} allocated: {}".format(gc.mem_free(), gc.mem_alloc()))
    gc.collect()
    print(
        "Garbage collect free: {} allocated: {}".format(gc.mem_free(), gc.mem_alloc())
    )
    print("-----------------------------")
    micropython.mem_info(1)
