import ast
import asyncio
import base64
import io
from langchain_core.tools import BaseTool
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from multiprocessing import Process, Queue
import psutil
from PIL import Image
from pydantic import BaseModel, Field
import time
from typing import Any, Mapping, Optional, Type


def interactive_run(
    code: str, globals: Optional[dict], locals: Optional[Mapping],
    type_ignores: list = [],
):
    # jupyter style interactive Python, the last expression is evaluated
    # for the result
    nodes = ast.parse(code)
    if len(nodes.body) == 0:
        return None

    if isinstance(nodes.body[-1], ast.Expr):
        exec_nodes = nodes.body[:-1]
        eval_nodes = nodes.body[-1:]
    else:
        exec_nodes = nodes.body
        eval_nodes = []

    if len(exec_nodes) > 0:
        exec_code = ast.unparse(ast.Module(exec_nodes, type_ignores))
        result = exec(exec_code, globals, locals)

    if len(eval_nodes) > 0:
        eval_code = ast.unparse(ast.Module(eval_nodes, type_ignores))
        result = eval(eval_code, globals, locals)

    return result


def append_content_text_to_content(content: list, text: str):
    if len(content) > 0 and content[-1]["type"] == "text":
        content[-1]["text"] += text
    else:
        content.append({"type": "text", "text": text})


def make_output_content(
    exec_out: io.StringIO, result: Any, image_save_kwargs: dict = {},
    print_image_object: bool = False,
):
    content = []
    if exec_out.tell() > 0:
        content.append({"type": "text", "text": exec_out.getvalue()})

    if result is not None:
        if isinstance(result, Image.Image):
            try:
                with io.BytesIO() as f:
                    result.save(f, **image_save_kwargs)
                    f.seek(0)
                    mime = Image.open(f).get_format_mimetype()
                    base64_data = base64.b64encode(f.getvalue())

                url = "data:{};base64,{}".format(
                    mime, base64_data.decode("utf-8"))
                content.append({
                    "type": "image_url", "image_url": {"url": url}
                })
                if print_image_object:
                    append_content_text_to_content(
                        content, str(result))

            except Exception as e:
                append_content_text_to_content(content, repr(e))

        else:
            result_str = result if isinstance(result, str) else str(result)
            append_content_text_to_content(content, result_str)

    return content


def simplify_content(content: list):
    if len(content) == 0:
        return ""
    elif len(content) == 1 and content[0]["type"] == "text":
        return content[0]["text"]
    else:
        return content


def interactive_python_process(
    input_queue: Queue, output_queue: Queue, globals={}, locals=None,
    image_save_kwargs: dict = {"format": "PNG"},
    print_image_object: bool = False, type_ignores: list = [],
):
    from contextlib import redirect_stdout

    # the process of execution loop gets code from the input queue, returns the
    # response content to the output queue, and maintains the state
    is_shutdown = False
    while not is_shutdown:
        task_name, task_args = input_queue.get()
        if task_name == "shutdown":
            is_shutdown = True
        elif task_name == "execute":
            with io.StringIO() as exec_out:
                # capture the standard output, last expression result, and
                # exception, for the tool response content
                result = None
                try:
                    with redirect_stdout(exec_out):
                        result = interactive_run(
                            task_args, globals, locals, type_ignores)
                except Exception as e:
                    exec_out.write(repr(e) + "\n")

                content = make_output_content(
                    exec_out, result, image_save_kwargs,
                    print_image_object,
                )

            output_queue.put(content)


class InteractivePythonReplInput(BaseModel):
    code: str = Field(description="Python code to execute")


class InteractivePythonRepl(BaseTool):

    args_schema: Type[BaseModel] = InteractivePythonReplInput
    name: str = "python_repl"
    description: str = (
        "Execute the Python code interactively, returning all printed text "
        "outputs and the result of the last expression (if any)."
    )

    globals: Optional[dict] = Field(default_factory=dict)
    """The global variables for exec() or eval()."""

    locals: Optional[Mapping] = None
    """The local variables for exec() or eval()."""

    image_save_kwargs: dict = {"format": "PNG"}
    """The options to save the image returned in the 'ipython' mode."""

    print_image_object: bool = False
    """Output images as text objects to the text content."""

    sessions: dict = {}
    """The process, queues, last modified time for each session indexed by the
    `metadata["thread_id"]`.
    """

    session_memory_limits: int = 4 * (2 ** 30)
    """The memory limits for the code session."""

    session_expiration_time_ns: int = 300_000000000
    """The expiration time in nanoseconds for the code session."""

    check_interval: float = 0.05
    """The interval to check if the running code should be killed due to
    timeout or out of memory limits.
    """

    @staticmethod
    def get_uss(meminfo):
        shared = getattr(meminfo, "shared", 0)
        return meminfo.rss - shared

    def clean_expired(self, now_ns: int):
        expired = [
            k for k, v in self.sessions.items()
            if (
                now_ns - v["modified_ns"] >= self.session_expiration_time_ns
            )
        ]
        for i in expired:
            session = self.sessions.pop(i)
            session["input_queue"].put(("shutdown", None))
            session["process"].join()

    def check_memory(self, process: Process):
        if process.pid is not None:
            parent = psutil.Process(process.pid)
            total_memory = parent.memory_info().rss
            for i in parent.children(recursive=True):
                total_memory += InteractivePythonRepl.get_uss(i.memory_info())

            if total_memory >= self.session_memory_limits:
                return True, "OutOfMemoryError"

        return False, None

    def get_or_set(self, key: str):
        if key not in self.sessions:
            input_queue = Queue()
            output_queue = Queue()
            process = Process(
                target=interactive_python_process,
                args=(
                    input_queue, output_queue, self.globals, self.locals,
                    self.image_save_kwargs, self.print_image_object,
                ),
                daemon=True,
            )
            self.sessions[key] = {
                "input_queue": input_queue,
                "output_queue": output_queue,
                "process": process,
            }
            process.start()

        result = self.sessions[key]
        op_time_ns = result["modified_ns"] = time.time_ns()
        self.clean_expired(op_time_ns)
        return result

    def _run(
        self, code: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ):
        assert "thread_id" in run_manager.metadata, \
            "The thread ID is required for interactive Python."
        thread_id = run_manager.metadata["thread_id"]
        session = self.get_or_set(thread_id)
        session["input_queue"].put(("execute", code))
        while session["output_queue"].empty():
            time.sleep(self.check_interval)
            should_kill, reason = self.check_memory(session["process"])
            if should_kill:
                self.sessions.pop(thread_id)["process"].kill()
                return reason

        return simplify_content(session["output_queue"].get(False))

    async def _arun(
        self, code: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ):
        assert "thread_id" in run_manager.metadata, \
            "The thread ID is required for interactive Python."
        thread_id = run_manager.metadata["thread_id"]
        session = self.get_or_set(thread_id)
        session["input_queue"].put(("execute", code))
        while session["output_queue"].empty():
            try:
                await asyncio.sleep(self.check_interval)
                should_kill, reason = self.check_memory(session["process"])
            except TimeoutError as e:
                should_kill = True
                reason = repr(e)

            if should_kill:
                self.sessions.pop(thread_id)["process"].kill()
                return reason

        return simplify_content(session["output_queue"].get(False))
