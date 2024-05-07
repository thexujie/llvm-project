// RUN: %clang_cc1 %s -ast-dump -fblocks | FileCheck %s

// Make sure that the attribute gets parsed and attached to the correct AST elements.

#pragma clang diagnostic ignored "-Wunused-variable"

// =========================================================================================
// Square brackets, true

namespace square_brackets {

// On the type of the FunctionDecl
void nl_function() [[clang::nonblocking]];
// CHECK: FunctionDecl {{.*}} nl_function 'void () __attribute__((clang_nonblocking))'

// On the type of the VarDecl holding a function pointer
void (*nl_func_a)() [[clang::nonblocking]];
// CHECK: VarDecl {{.*}} nl_func_a 'void (*)() __attribute__((clang_nonblocking))'

// On the type of the ParmVarDecl of a function parameter
static void nlReceiver(void (*nl_func)() [[clang::nonblocking]]);
// CHECK: ParmVarDecl {{.*}} nl_func 'void (*)() __attribute__((clang_nonblocking))'

// As an AttributedType within the nested types of a typedef
typedef void (*nl_fp_type)() [[clang::nonblocking]];
// CHECK: TypedefDecl {{.*}} nl_fp_type 'void (*)() __attribute__((clang_nonblocking))'
using nl_fp_talias = void (*)() [[clang::nonblocking]];
// CHECK: TypeAliasDecl {{.*}} nl_fp_talias 'void (*)() __attribute__((clang_nonblocking))'

// From a typedef or typealias, on a VarDecl
nl_fp_type nl_fp_var1;
// CHECK: VarDecl {{.*}} nl_fp_var1 'nl_fp_type':'void (*)() __attribute__((clang_nonblocking))'
nl_fp_talias nl_fp_var2;
// CHECK: VarDecl {{.*}} nl_fp_var2 'nl_fp_talias':'void (*)() __attribute__((clang_nonblocking))'

// On type of a FieldDecl
struct Struct {
	void (*nl_func_field)() [[clang::nonblocking]];
// CHECK: FieldDecl {{.*}} nl_func_field 'void (*)() __attribute__((clang_nonblocking))'
};

// nonallocating should NOT be subsumed into nonblocking
void nl1() [[clang::nonblocking]] [[clang::nonallocating]];
// CHECK: FunctionDecl {{.*}} nl1 'void () __attribute__((clang_nonblocking)) __attribute__((clang_nonallocating))'

void nl2() [[clang::nonallocating]] [[clang::nonblocking]];
// CHECK: FunctionDecl {{.*}} nl2 'void () __attribute__((clang_nonblocking)) __attribute__((clang_nonallocating))'

decltype(nl1) nl3;
// CHECK: FunctionDecl {{.*}} nl3 'decltype(nl1)':'void () __attribute__((clang_nonblocking)) __attribute__((clang_nonallocating))'

// Attribute propagates from base class virtual method to overrides.
struct Base {
	virtual void nb_method() [[clang::nonblocking]];
};
struct Derived : public Base {
	void nb_method() override;
	// CHECK: CXXMethodDecl {{.*}} nb_method 'void () __attribute__((clang_nonblocking))'
};

// Dependent expression
template <bool V>
struct Dependent {
	void nb_method2() [[clang::nonblocking(V)]];
	// CHECK: CXXMethodDecl {{.*}} nb_method2 'void () __attribute__((clang_nonblocking(V)))'
};

// --- Blocks ---

// On the type of the VarDecl holding a BlockDecl
void (^nl_block1)() [[clang::nonblocking]] = ^() [[clang::nonblocking]] {};
// CHECK: VarDecl {{.*}} nl_block1 'void (^)() __attribute__((clang_nonblocking))'

int (^nl_block2)() [[clang::nonblocking]] = ^() [[clang::nonblocking]] { return 0; };
// CHECK: VarDecl {{.*}} nl_block2 'int (^)() __attribute__((clang_nonblocking))'

// The operand of the CallExpr is an ImplicitCastExpr of a DeclRefExpr -> nl_block which hold the attribute
static void blockCaller() { nl_block1(); }
// CHECK: DeclRefExpr {{.*}} 'nl_block1' 'void (^)() __attribute__((clang_nonblocking))'

// --- Lambdas ---

// On the operator() of a lambda's CXXMethodDecl
auto nl_lambda = []() [[clang::nonblocking]] {};
// CHECK: CXXMethodDecl {{.*}} operator() 'void () const __attribute__((clang_nonblocking))' inline

// =========================================================================================
// Square brackets, false

void nl_func_false() [[clang::blocking]];
// CHECK: FunctionDecl {{.*}} nl_func_false 'void () __attribute__((clang_blocking))'

// TODO: This exposes a bug where a type attribute is lost when inferring a lambda's
// return type.
auto nl_lambda_false = []() [[clang::blocking]] {};

} // namespace square_brackets

// =========================================================================================
// GNU-style attribute, true

// TODO: Duplicate more of the above for GNU-style attribute

namespace gnu_style {

// On the type of the FunctionDecl
void nl_function() __attribute__((clang_nonblocking));
// CHECK: FunctionDecl {{.*}} nl_function 'void () __attribute__((clang_nonblocking))'

// Alternate placement on the FunctionDecl
__attribute__((clang_nonblocking)) void nl_function();
// CHECK: FunctionDecl {{.*}} nl_function 'void () __attribute__((clang_nonblocking))'

// On the type of the VarDecl holding a function pointer
void (*nl_func_a)() __attribute__((clang_nonblocking));
// CHECK: VarDecl {{.*}} nl_func_a 'void (*)() __attribute__((clang_nonblocking))'

// Alternate attribute placement on VarDecl
__attribute__((clang_nonblocking)) void (*nl_func_b)();
// CHECK: VarDecl {{.*}} nl_func_b 'void (*)() __attribute__((clang_nonblocking))'

} // namespace gnu_style

// TODO: Duplicate the above for nonallocating




// =========================================================================================
// Non-blocking with an expression parameter

void t0() [[clang::nonblocking(1 - 1)]];
// CHECK: FunctionDecl {{.*}} t0 'void () __attribute__((clang_blocking))'
void t1() [[clang::nonblocking(1 + 1)]];
// CHECK: FunctionDecl {{.*}} t1 'void () __attribute__((clang_nonblocking))'

template <bool V>
struct ValueDependent {
	void nb_method() [[clang::nonblocking(V)]];
};

void t3() [[clang::nonblocking]]
{
	ValueDependent<false> x1;
	x1.nb_method();
// CHECK: ClassTemplateSpecializationDecl {{.*}} ValueDependent
// CHECK: TemplateArgument integral 0
// CHECK: CXXMethodDecl {{.*}} nb_method 'void () __attribute__((clang_blocking))'

 	ValueDependent<true> x2;
 	x2.nb_method();
// CHECK: ClassTemplateSpecializationDecl {{.*}} ValueDependent
// CHECK: TemplateArgument integral 1
// CHECK: CXXMethodDecl {{.*}} nb_method 'void () __attribute__((clang_nonblocking))'
}

template <typename X>
struct TypeDependent {
	void td_method() [[clang::nonblocking(X::is_nb)]];
};

struct NBPolicyTrue {
	static constexpr bool is_nb = true;
};

struct NBPolicyFalse {
	static constexpr bool is_nb = false;
};

void t4()
{
	TypeDependent<NBPolicyFalse> x1;
	x1.td_method();
// CHECK: ClassTemplateSpecializationDecl {{.*}} TypeDependent
// CHECK: TemplateArgument type 'NBPolicyFalse'
// CHECK: CXXMethodDecl {{.*}} td_method 'void () __attribute__((clang_blocking))'

	TypeDependent<NBPolicyTrue> x2;
	x2.td_method();
// CHECK: ClassTemplateSpecializationDecl {{.*}} TypeDependent
// CHECK: TemplateArgument type 'NBPolicyTrue'
// CHECK: CXXMethodDecl {{.*}} td_method 'void () __attribute__((clang_nonblocking))'
}

