import torch


def get_disc_loss(real, fake, disc, disc_criterion):
    output = disc(fake.detach())
    disc_fake_loss = disc_criterion(output, torch.zeros_like(output))
    output = disc(real)
    disc_real_loss = disc_criterion(output, torch.ones_like(output))
    disc_loss = (disc_fake_loss + disc_real_loss) / 2
    return disc_loss


def get_adversarial_loss(real_X, disc_Y, gen_XY, adv_criterion):
    fake_Y = gen_XY(real_X)
    output = disc_Y(fake_Y)
    adversarial_loss = adv_criterion(output, torch.ones_like(output))
    return adversarial_loss, fake_Y


def get_identity_loss(real_X, gen_YX, identity_criterion):
    identity_X = gen_YX(real_X)
    identity_loss = identity_criterion(real_X, identity_X)
    return identity_loss


def get_cycle_consistency_loss(real_X, fake_Y, gen_YX, cycle_criterion):
    cycle_X = gen_YX(fake_Y)
    cycle_loss = cycle_criterion(cycle_X, real_X)
    return cycle_loss


def get_gen_loss(
        real_X, real_Y, gen_XY, gen_YX, disc_X, disc_Y, adv_criterion,
        identity_criterion, cycle_criterion, lambda_identity=0.1, lambda_cycle=10):
    adv_loss_XY, fake_Y = get_adversarial_loss(real_X, disc_Y, gen_XY, adv_criterion)
    adv_loss_YX, fake_X = get_adversarial_loss(real_Y, disc_X, gen_XY, adv_criterion)
    adv_loss = adv_loss_XY + adv_loss_YX

    idt_loss_XY = get_identity_loss(real_X, gen_YX, identity_criterion)
    idt_loss_YX = get_identity_loss(real_Y, gen_XY, identity_criterion)
    idt_loss = idt_loss_XY + idt_loss_YX

    cyc_loss_XY = get_cycle_consistency_loss(real_X, fake_Y, gen_YX, cycle_criterion)
    cyc_loss_YX = get_cycle_consistency_loss(real_Y, fake_X, gen_XY, cycle_criterion)
    cyc_loss = cyc_loss_YX + cyc_loss_XY

    gen_loss = adv_loss + lambda_identity*idt_loss + lambda_cycle*cyc_loss
    return gen_loss, fake_X, fake_Y
