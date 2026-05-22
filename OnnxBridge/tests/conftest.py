import pytest
import tempfile
import shutil
import os
import sys


def pytest_addoption(parser):
    parser.addoption(
        "--backend",
        action="store",
        choices=[
            "CLEARTEXT_LLAMA",
            "LLAMA",
            "SECFLOAT",
            "SECFLOAT_CLEARTEXT",
            "CLEARTEXT_fp",
        ],
        help="backend : CLEARTEXT_LLAMA | CLEARTEXT_fp | LLAMA | SECFLOAT | SECFLOAT_CLEARTEXT",
        required=True,
    )
    parser.addoption(
        "--batch_size",
        action="store",
        type=int,
        help="batch size",
        required=False,
    )
    parser.addoption(
        "--model",
        action="store",
        help="absolute mdel path",
        required=False,
    )
    parser.addoption(
        "--input_name",
        action="store",
        help="absolute input_name path",
        required=False,
    )


@pytest.fixture(scope="session")
def backend(request):
    opt = request.config.getoption("--backend")
    return opt


@pytest.fixture(scope="session")
def model(request):
    opt = request.config.getoption("--model")
    return opt


@pytest.fixture(scope="session")
def input_name(request):
    opt = request.config.getoption("--input_name")
    return opt


@pytest.fixture(scope="session")
def batch_size(request):
    opt = request.config.getoption("--batch_size")
    return opt


@pytest.fixture(scope="session", autouse=True)
def test_env():
    config = {}
    # Get the directory path where the current script is located
    script_directory = os.path.dirname(os.path.abspath(__file__))

    test_dir = "onnxBridge_tests"
    path = os.path.join(script_directory, test_dir)
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=False)
    os.listdir()
    os.mkdir(path)
    config["test_dir"] = path
    return config


def make_dir(path):
    print(path)
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=False)
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
    print("\nRequest node: ", request.node.name)
    if "[" in request.node.name:
        test_name_list = request.node.name.split("[")
        parameter_name = test_name_list[1].split("]")[0]
    else:
        test_name_list = request.node.name.split("_")
        parameter_name = "custom"
    full_test_name = test_name_list[0] + "_" + parameter_name
    test_name = full_test_name[len("test_") :]
    main_test_dir = test_env["test_dir"]
    print("Main test dir: ", main_test_dir)
    test_dir = os.path.join(main_test_dir, "test_" + test_name)
    make_dir(test_dir)

    yield test_dir
    # print("Test dir: ", test_dir)
    # Remove dir only if test passed
    if not request.node.rep_call.failed:
        shutil.rmtree(test_dir, ignore_errors=False)
    return
