import os
import re
import sys
import asyncio
from aiohttp import web, ClientSession

MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENCY", "512"))
SCHEDULER_TIMEOUT = int(os.getenv("SCHEDULER_TIMEOUT", "60"))
TIMEOUTS = [SCHEDULER_TIMEOUT]


def extract_ips(file_path, world_size):
    # 定义匹配IP地址的正则表达式
    ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
    ips = []
    
    with open(file_path, 'r') as file:
        # 读取文件所有行
        lines = file.readlines()
        
        # 提取倒数 WORLD_SIZE 行
        lines_to_check = lines[-world_size:]
        
        # 连接倒数的几行成一段文本
        content_after_marker = ''.join(lines_to_check)
        
        # 提取 IP 地址
        ips = re.findall(ip_pattern, content_after_marker)
    
    return ips


class Scheduler:
    def __init__(self, servers, max_concurrent):
        self.servers = servers
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.next_server = 0
        self.lock = asyncio.Lock()

    async def get_server(self):
        async with self.lock:
            server = self.servers[self.next_server]
            self.next_server = (self.next_server + 1) % len(self.servers)
            return server

    async def forward(self, request, timeout):
        backend = await self.get_server()
        url = f"{backend}{request.rel_url}"

        try:
            async with ClientSession() as session:
                async with session.request(
                    method=request.method,
                    url=url,
                    headers={k: v for k, v in request.headers.items() if k.lower() != 'host'},
                    data=await request.read(),
                    timeout=timeout
                ) as resp:
                    body = await resp.read()
                    return web.Response(body=body, status=resp.status, headers=resp.headers)
        except asyncio.TimeoutError:
            return None  # 表示超时
        except Exception as e:
            return web.Response(status=502, text=f"Upstream error: {str(e)}")

    async def proxy(self, request):
        data = await request.read()
        headers = dict(request.headers)

        # 用于重试，不再重复读取request
        async def make_retry_req():
            # 构造一个新request-like对象，带原始body和headers
            class DummyReq:
                def __init__(self, rel_url, method, data, headers):
                    self.rel_url = rel_url
                    self.method = method
                    self._data = data
                    self.headers = headers
                async def read(self): return self._data
            return DummyReq(request.rel_url, request.method, data, headers)

        # 3级重试调度
        for i, timeout in enumerate(TIMEOUTS):
            async with self.semaphore:
                retry_req = await make_retry_req() if i > 0 else request
                resp = await self.forward(retry_req, timeout)
            if resp is not None:
                return resp  # 成功或其它异常
            # 超时则进入下一级重试
        return web.Response(status=504, text=f"Request timed out after {sum(TIMEOUTS)}s")


NGINX_CONF_PATH = '/etc/nginx/nginx.conf'
HOSTS_FILE = '/etc/hosts'
PORT=8080

# 提取IP
world_size = int(os.getenv('WORLD_SIZE', 1))
if world_size == 1:
   ips = ["127.0.0.1"]
else:
   ips = extract_ips(HOSTS_FILE, world_size)
BACKEND_SERVERS = [f"http://{ip}:{PORT}" for ip in ips]


# BACKEND_SERVERS = sys.argv[1].split(',')
print(BACKEND_SERVERS)


# aiohttp app
scheduler = Scheduler(BACKEND_SERVERS, MAX_CONCURRENT)
app = web.Application(client_max_size=2*1024*1024)
app.router.add_route('*', '/{tail:.*}', scheduler.proxy)

if __name__ == "__main__":
    web.run_app(app, host='0.0.0.0', port=80)
