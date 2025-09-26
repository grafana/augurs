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

type OutlierInterval struct {
	Start uint32
	End uint32
}

type Series struct {
	IsOutlier bool
	OutlierIntervals []OutlierInterval
	Scores []float64
}

type Band struct {
	Min []float64
	Max []float64
}

type Output struct {
	OutlyingSeries []uint32
	SeriesResults []Series
	ClusterBand Band, bool
}

type Error = string

type DbscanParams struct {
	EpsilonOrSensitivity interface{}
}

type MadParams struct {
	ThresholdOrSensitivity interface{}
}

type Input struct {
	Data [][]float64
	Algorithm interface{}
}

type IOutlierTypes interface {}

func NewOutlierFactory(
	ctx context.Context,
	types IOutlierTypes,
) (*OutlierFactory, error) {
	wazeroRuntime := wazero.NewRuntime(ctx)

	_, err0 := wazeroRuntime.NewHostModuleBuilder("augurs:outlier/types").
	Instantiate(ctx)
	if err0 != nil {
		return nil, err0
	}

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
	input Input,
) (Output, error) {
	arg0 := input
	data0 := arg0.Data
	algorithm0 := arg0.Algorithm
	vec3 := data0
	len3 := uint64(len(vec3))
	result3, err3 := i.module.ExportedFunction("cabi_realloc").Call(ctx, 0, 0, 4, len3 * 8)
	if err3 != nil {
		var default3 Output
		return default3, err3
	}
	ptr3 := result3[0]
	for idx := uint64(0); idx < len3; idx++ {
		e := vec3[idx]
		base := uint32(ptr3 + uint64(idx) * uint64(8))
		vec2 := e
		len2 := uint64(len(vec2))
		result2, err2 := i.module.ExportedFunction("cabi_realloc").Call(ctx, 0, 0, 8, len2 * 8)
		if err2 != nil {
			var default2 Output
			return default2, err2
		}
		ptr2 := result2[0]
		for idx := uint64(0); idx < len2; idx++ {
			e := vec2[idx]
			base := uint32(ptr2 + uint64(idx) * uint64(8))
			result1 := api.EncodeF64(e)
			i.module.Memory().WriteUint64Le(base+0, result1)
		}
		i.module.Memory().WriteUint32Le(base+4, uint32(len2))
		i.module.Memory().WriteUint32Le(base+0, uint32(ptr2))
	}
	var variant12_0 uint32
	var variant12_1 uint32
	var variant12_2 float64
	switch variantPayload := algorithm0.(type) {
		case Dbscan:
			epsilonOrSensitivity4 := variantPayload.EpsilonOrSensitivity
			var variant7_0 uint32
			var variant7_1 float64
			switch variantPayload := epsilonOrSensitivity4.(type) {
				case Sensitivity:
					result5 := api.EncodeF64(variantPayload)
					variant7_0 = 0
					variant7_1 = result5
				case Epsilon:
					result6 := api.EncodeF64(variantPayload)
					variant7_0 = 1
					variant7_1 = result6
				default:
					var default7 Output
					return default7, errors.New("invalid variant type provided")
			}
			variant12_0 = 0
			variant12_1 = variant7_0
			variant12_2 = variant7_1
		case Mad:
			thresholdOrSensitivity8 := variantPayload.ThresholdOrSensitivity
			var variant11_0 uint32
			var variant11_1 float64
			switch variantPayload := thresholdOrSensitivity8.(type) {
				case Sensitivity:
					result9 := api.EncodeF64(variantPayload)
					variant11_0 = 0
					variant11_1 = result9
				case Threshold:
					result10 := api.EncodeF64(variantPayload)
					variant11_0 = 1
					variant11_1 = result10
				default:
					var default11 Output
					return default11, errors.New("invalid variant type provided")
			}
			variant12_0 = 1
			variant12_1 = variant11_0
			variant12_2 = variant11_1
		default:
			var default12 Output
			return default12, errors.New("invalid variant type provided")
	}
	raw13, err13 := i.module.ExportedFunction("detect").Call(ctx, uint64(ptr3), uint64(len3), uint64(variant12_0), uint64(variant12_1), uint64(variant12_2))
	if err13 != nil {
		var default13 Output
		return default13, err13
	}

	// The cleanup via `cabi_post_*` cleans up the memory in the guest. By
	// deferring this, we ensure that no memory is corrupted before the function
	// is done accessing it.
	defer func() {
		if _, err := i.module.ExportedFunction("cabi_post_detect").Call(ctx, raw13...); err != nil {
			// If we get an error during cleanup, something really bad is
			// going on, so we panic. Also, you can't return the error from
			// the `defer`
			panic(errors.New("failed to cleanup"))
		}
	}()

	results13 := raw13[0]
	value14, ok14 := i.module.Memory().ReadByte(uint32(results13 + 0))
	if !ok14 {
		var default14 Output
		return default14, errors.New("failed to read byte from memory")
	}
	var value58 Output
	var err58 error
	switch value14 {
	case 0:
		ptr15, ok15 := i.module.Memory().ReadUint32Le(uint32(results13 + 4))
		if !ok15 {
			var default15 Output
			return default15, errors.New("failed to read pointer from memory")
		}
		len16, ok16 := i.module.Memory().ReadUint32Le(uint32(results13 + 8))
		if !ok16 {
			var default16 Output
			return default16, errors.New("failed to read length from memory")
		}
		base19 := ptr15
		len19 := len16
		result19 := make([]uint32, len19)
		for idx19 := uint32(0); idx19 < len19; idx19++ {
			base := base19 + idx19 * 4
			value17, ok17 := i.module.Memory().ReadUint32Le(uint32(base + 0))
			if !ok17 {
				var default17 Output
				return default17, errors.New("failed to read i32 from memory")
			}
			result18 := api.DecodeU32(uint64(value17))
			result19[idx19] = result18
		}
		ptr20, ok20 := i.module.Memory().ReadUint32Le(uint32(results13 + 12))
		if !ok20 {
			var default20 Output
			return default20, errors.New("failed to read pointer from memory")
		}
		len21, ok21 := i.module.Memory().ReadUint32Le(uint32(results13 + 16))
		if !ok21 {
			var default21 Output
			return default21, errors.New("failed to read length from memory")
		}
		base40 := ptr20
		len40 := len21
		result40 := make([]Series, len40)
		for idx40 := uint32(0); idx40 < len40; idx40++ {
			base := base40 + idx40 * 20
			value22, ok22 := i.module.Memory().ReadByte(uint32(base + 0))
			if !ok22 {
				var default22 Output
				return default22, errors.New("failed to read byte from memory")
			}
			value23 := value22 != 0
			ptr24, ok24 := i.module.Memory().ReadUint32Le(uint32(base + 4))
			if !ok24 {
				var default24 Output
				return default24, errors.New("failed to read pointer from memory")
			}
			len25, ok25 := i.module.Memory().ReadUint32Le(uint32(base + 8))
			if !ok25 {
				var default25 Output
				return default25, errors.New("failed to read length from memory")
			}
			base33 := ptr24
			len33 := len25
			result33 := make([]OutlierInterval, len33)
			for idx33 := uint32(0); idx33 < len33; idx33++ {
				base := base33 + idx33 * 12
				value26, ok26 := i.module.Memory().ReadUint32Le(uint32(base + 0))
				if !ok26 {
					var default26 Output
					return default26, errors.New("failed to read i32 from memory")
				}
				result27 := api.DecodeU32(uint64(value26))
				value28, ok28 := i.module.Memory().ReadByte(uint32(base + 4))
				if !ok28 {
					var default28 Output
					return default28, errors.New("failed to read byte from memory")
				}
				var result31 uint32
				var ok31 bool
				if value28 == 0 {
					ok31 = false
				} else {
					value29, ok29 := i.module.Memory().ReadUint32Le(uint32(base + 8))
					if !ok29 {
						var default29 Output
						return default29, errors.New("failed to read i32 from memory")
					}
					result30 := api.DecodeU32(uint64(value29))
					ok31 = true
					result31 = result30
				}
				value32 := OutlierInterval{
					Start: result27,
					End: result31, ok31,
				}
				result33[idx33] = value32
			}
			ptr34, ok34 := i.module.Memory().ReadUint32Le(uint32(base + 12))
			if !ok34 {
				var default34 Output
				return default34, errors.New("failed to read pointer from memory")
			}
			len35, ok35 := i.module.Memory().ReadUint32Le(uint32(base + 16))
			if !ok35 {
				var default35 Output
				return default35, errors.New("failed to read length from memory")
			}
			base38 := ptr34
			len38 := len35
			result38 := make([]float64, len38)
			for idx38 := uint32(0); idx38 < len38; idx38++ {
				base := base38 + idx38 * 8
				value36, ok36 := i.module.Memory().ReadUint64Le(uint32(base + 0))
				if !ok36 {
					var default36 Output
					return default36, errors.New("failed to read f64 from memory")
				}
				result37 := api.DecodeF64(value36)
				result38[idx38] = result37
			}
			value39 := Series{
				IsOutlier: value23,
				OutlierIntervals: result33,
				Scores: result38,
			}
			result40[idx40] = value39
		}
		value41, ok41 := i.module.Memory().ReadByte(uint32(results13 + 20))
		if !ok41 {
			var default41 Output
			return default41, errors.New("failed to read byte from memory")
		}
		var result53 Band
		var ok53 bool
		if value41 == 0 {
			ok53 = false
		} else {
			ptr42, ok42 := i.module.Memory().ReadUint32Le(uint32(results13 + 24))
			if !ok42 {
				var default42 Output
				return default42, errors.New("failed to read pointer from memory")
			}
			len43, ok43 := i.module.Memory().ReadUint32Le(uint32(results13 + 28))
			if !ok43 {
				var default43 Output
				return default43, errors.New("failed to read length from memory")
			}
			base46 := ptr42
			len46 := len43
			result46 := make([]float64, len46)
			for idx46 := uint32(0); idx46 < len46; idx46++ {
				base := base46 + idx46 * 8
				value44, ok44 := i.module.Memory().ReadUint64Le(uint32(base + 0))
				if !ok44 {
					var default44 Output
					return default44, errors.New("failed to read f64 from memory")
				}
				result45 := api.DecodeF64(value44)
				result46[idx46] = result45
			}
			ptr47, ok47 := i.module.Memory().ReadUint32Le(uint32(results13 + 32))
			if !ok47 {
				var default47 Output
				return default47, errors.New("failed to read pointer from memory")
			}
			len48, ok48 := i.module.Memory().ReadUint32Le(uint32(results13 + 36))
			if !ok48 {
				var default48 Output
				return default48, errors.New("failed to read length from memory")
			}
			base51 := ptr47
			len51 := len48
			result51 := make([]float64, len51)
			for idx51 := uint32(0); idx51 < len51; idx51++ {
				base := base51 + idx51 * 8
				value49, ok49 := i.module.Memory().ReadUint64Le(uint32(base + 0))
				if !ok49 {
					var default49 Output
					return default49, errors.New("failed to read f64 from memory")
				}
				result50 := api.DecodeF64(value49)
				result51[idx51] = result50
			}
			value52 := Band{
				Min: result46,
				Max: result51,
			}
			ok53 = true
			result53 = value52
		}
		value54 := Output{
			OutlyingSeries: result19,
			SeriesResults: result40,
			ClusterBand: result53, ok53,
		}
		value58 = value54
	case 1:
		ptr55, ok55 := i.module.Memory().ReadUint32Le(uint32(results13 + 4))
		if !ok55 {
			var default55 Output
			return default55, errors.New("failed to read pointer from memory")
		}
		len56, ok56 := i.module.Memory().ReadUint32Le(uint32(results13 + 8))
		if !ok56 {
			var default56 Output
			return default56, errors.New("failed to read length from memory")
		}
		buf57, ok57 := i.module.Memory().Read(ptr55, len56)
		if !ok57 {
			var default57 Output
			return default57, errors.New("failed to read bytes from memory")
		}
		str57 := string(buf57)
		err58 = errors.New(str57)
	default:
		err58 = errors.New("invalid variant discriminant for expected")
	}
	return value58, err58
}
