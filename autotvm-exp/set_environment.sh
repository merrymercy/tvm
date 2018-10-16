python3 -m tvm.exec.rpc_tracker &
python3 -m tvm.exec.rpc_server --tracker localhost:9190 --key titanx &

python3 -m tvm.exec.query_rpc_tracker
