package outlier

import "context"
import "errors"
import "github.com/tetratelabs/wazero"
import "github.com/tetratelabs/wazero/api"

import _ "embed"

//go:embed outlier.wasm
var wasmFileOutlier []byte

type OutlierFactory struct {
	runtime wazero.Runtime
	module wazero.CompiledModule
}

func NewOutlierFactory(
	ctx context.Context,
) (*OutlierFactory, error) {
	wazeroRuntime := wazero.NewRuntime(ctx)

	// Compiling the module takes a LONG time, so we want to do it once and hold
	// onto it with the Runtime
	module, err := wazeroRuntime.CompileModule(ctx, wasmFileOutlier)
	if err != nil {
		return nil, err
	}

	return &OutlierFactory{wazeroRuntime, module}, nil
}

func (f *OutlierFactory) Instantiate(ctx context.Context) (*OutlierInstance, error) {
	if module, err := f.runtime.InstantiateModule(ctx, f.module, wazero.NewModuleConfig()); err != nil {
		return nil, err
	} else {
		return &OutlierInstance{module}, nil
	}
}

func (f *OutlierFactory) Close(ctx context.Context) {
	f.runtime.Close(ctx)
}

type OutlierInstance struct {
	module api.Module
}

// writeString will put a Go string into the Wasm memory following the Component
// Model calling convetions, such as allocating memory with the realloc function
func writeString(
	ctx context.Context,
	s string,
	memory api.Memory,
	realloc api.Function,
) (uint64, uint64, error) {
	if len(s) == 0 {
		return 1, 0, nil
	}

	results, err := realloc.Call(ctx, 0, 0, 1, uint64(len(s)))
	if err != nil {
		return 1, 0, err
	}
	ptr := results[0]
	ok := memory.Write(uint32(ptr), []byte(s))
	if !ok {
		return 1, 0, err
	}
	return uint64(ptr), uint64(len(s)), nil
}

func (i *OutlierInstance) Close(ctx context.Context) error {
	if err := i.module.Close(ctx); err != nil {
		return err
	}

	return nil
}

func (i *OutlierInstance) Detect(
	ctx context.Context,
	input string,
) (string, error) {
	arg0 := input
	memory0 := i.module.Memory()
	realloc0 := i.module.ExportedFunction("cabi_realloc")
	ptr0, len0, err0 := writeString(ctx, arg0, memory0, realloc0)
	if err0 != nil {
		var default0 string
		return default0, err0
	}
	raw1, err1 := i.module.ExportedFunction("detect").Call(ctx, uint64(ptr0), uint64(len0))
	if err1 != nil {
		var default1 string
		return default1, err1
	}

	// The cleanup via `cabi_post_*` cleans up the memory in the guest. By
	// deferring this, we ensure that no memory is corrupted before the function
	// is done accessing it.
	defer func() {
		if _, err := i.module.ExportedFunction("cabi_post_detect").Call(ctx, raw1...); err != nil {
			// If we get an error during cleanup, something really bad is
			// going on, so we panic. Also, you can't return the error from
			// the `defer`
			panic(errors.New("failed to cleanup"))
		}
	}()

	results1 := raw1[0]
	value2, ok2 := i.module.Memory().ReadByte(uint32(results1 + 0))
	if !ok2 {
		var default2 string
		return default2, errors.New("failed to read byte from memory")
	}
	var value9 string
	var err9 error
	switch value2 {
	case 0:
		ptr3, ok3 := i.module.Memory().ReadUint32Le(uint32(results1 + 4))
		if !ok3 {
			var default3 string
			return default3, errors.New("failed to read pointer from memory")
		}
		len4, ok4 := i.module.Memory().ReadUint32Le(uint32(results1 + 8))
		if !ok4 {
			var default4 string
			return default4, errors.New("failed to read length from memory")
		}
		buf5, ok5 := i.module.Memory().Read(ptr3, len4)
		if !ok5 {
			var default5 string
			return default5, errors.New("failed to read bytes from memory")
		}
		str5 := string(buf5)
		value9 = str5
	case 1:
		ptr6, ok6 := i.module.Memory().ReadUint32Le(uint32(results1 + 4))
		if !ok6 {
			var default6 string
			return default6, errors.New("failed to read pointer from memory")
		}
		len7, ok7 := i.module.Memory().ReadUint32Le(uint32(results1 + 8))
		if !ok7 {
			var default7 string
			return default7, errors.New("failed to read length from memory")
		}
		buf8, ok8 := i.module.Memory().Read(ptr6, len7)
		if !ok8 {
			var default8 string
			return default8, errors.New("failed to read bytes from memory")
		}
		str8 := string(buf8)
		err9 = errors.New(str8)
	default:
		err9 = errors.New("invalid variant discriminant for expected")
	}
	return value9, err9
}
