import pytest
import tempfile
import shutil
import os


def pytest_addoption(parser):
    parser.addoption(
        "--backend",
        action="store",
        default="CPP",
        help="backend : CPP | 2PC_HE | 2PC_OT | 3PC",
    )


@pytest.fixture(scope="session")
def backend(request):
    opt = request.config.getoption("--backend")
    if opt not in ["CPP", "3PC", "2PC_HE", "2PC_OT"]:
        opt = "CPP"
    return opt


@pytest.fixture(scope="session", autouse=True)
def test_env():
    config = {}
    test_dir = "cryptflow_tests"
    path = os.path.join(tempfile.gettempdir(), test_dir)
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)
    os.mkdir(path)
    config["test_dir"] = path
    return config


def make_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)
    else:
        os.mkdir(path)
    return


# Hook to check if test failed
@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    # execute all other hooks to obtain the report object
    outcome = yield
    rep = outcome.get_result()
    # set a report attribute for each phase of a call, which can
    # be "setup", "call", "teardown"
    setattr(item, "rep_" + rep.when, rep)


@pytest.fixture
def test_dir(request, test_env):
    test_name = request.node.name[len("test_") :]
    main_test_dir = test_env["test_dir"]
    test_dir = os.path.join(main_test_dir, "athos_test_" + test_name)
    make_dir(test_dir)
    yield test_dir
    # Remove dir only if test passed
    if not request.node.rep_call.failed:
        shutil.rmtree(test_dir, ignore_errors=True)
    return
