# %cd ~/dev/lrpc/gen_lrpc
# import xdbg

#%%
import sys
from google.protobuf.compiler import plugin_pb2

data = sys.stdin.buffer.read()
req = plugin_pb2.CodeGeneratorRequest.FromString(data)

resp = plugin_pb2.CodeGeneratorResponse()


SERVICER_TEMPLATE = """
class {service.name}Servicer:
  def _get_manifest(self):
    return [
%%SPLIT%%
    ]
"""

STUB_TEMPLATE = """
class {service.name}:
  def __init__(self, lrpc):
    manifest = [
%%SPLIT%%
    ]
    lrpc.add_stub(self, manifest)
"""

resp = plugin_pb2.CodeGeneratorResponse()

for proto_file in req.proto_file:
    python_file_name = proto_file.name.rstrip(".proto") + "_pb2.py"
    for service in proto_file.service:
        manifest_lines = []
        method_lines = []
        for method in service.method:
            input_type = method.input_type
            if input_type.startswith("."):
                input_type = input_type[1:]
            if input_type.startswith(proto_file.package):
                input_type = input_type.lstrip(proto_file.package)

            output_type = method.output_type
            if output_type.startswith("."):
                output_type = output_type[1:]
            if output_type.startswith(proto_file.package):
                output_type = output_type.lstrip(proto_file.package)

            line = "      ('{namespace}.{method.name}', self.{method.name}, {input_type}, {output_type}, {method.client_streaming}, {method.server_streaming}),".format(
                namespace=service.name,
                input_type=input_type,
                output_type=output_type,
                method=method
            )
            manifest_lines.append(line)

            if method.client_streaming:
                raise NotImplementedError("Client streaming is not implemented")

            if method.server_streaming:
                line_fmt = "  async def {method.name}(self, req, out):"
            else:
                line_fmt = "  async def {method.name}(self, req):"

            method_lines.append("")
            method_lines.append(line_fmt.format(method=method))
            method_lines.append("    raise NotImplementedError()")

        entry = resp.file.add()
        entry.name = python_file_name
        entry.insertion_point = 'module_scope'

        sections = SERVICER_TEMPLATE.format(service=service).split("%%SPLIT%%")
        entry.content = sections[0] + "\n".join(manifest_lines) + sections[1] + "\n".join(method_lines)

    for service in proto_file.service:
        manifest_lines = []
        for method in service.method:
            input_type = method.input_type
            if input_type.startswith("."):
                input_type = input_type[1:]
            if input_type.startswith(proto_file.package):
                input_type = input_type.lstrip(proto_file.package)

            output_type = method.output_type
            if output_type.startswith("."):
                output_type = output_type[1:]
            if output_type.startswith(proto_file.package):
                output_type = output_type.lstrip(proto_file.package)

            line = "      ('{namespace}.{method.name}', '{method.name}', {input_type}, {output_type}, {method.client_streaming}, {method.server_streaming}),".format(
                namespace=service.name,
                input_type=input_type,
                output_type=output_type,
                method=method
            )
            manifest_lines.append(line)

        entry = resp.file.add()
        entry.name = python_file_name
        entry.insertion_point = 'module_scope'

        sections = STUB_TEMPLATE.format(service=service).split("%%SPLIT%%")
        entry.content = sections[0] + "\n".join(manifest_lines) + sections[1]

sys.stdout.buffer.write(resp.SerializeToString())


# with open('gen_lrpc/example_data.bin', 'wb') as f:
#     f.write(data)

# with open('example_data.bin', 'rb') as f:
#     data = f.read()
